"""Executable detection-first research pipeline with independent calibration."""
from __future__ import annotations

import copy
import hashlib
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from .config import load_config, resolve_path, save_json
from .data import load_dataset
from .models import clamp_binary_master_weights
from .rbc_model import DetectionFirstBNN, normalized_detection_loss, conditional_type_loss
from .rbc_split import four_way_split, split_manifest


def _batches(size, batch_size, rng=None):
    order = None if rng is None else rng.permutation(size)
    # Merge the singleton tail instead of dropping an original row or breaking BN.
    for start in range(0, size, batch_size):
        if start and size - start == 1:
            break
        end = min(start + batch_size, size)
        if size - end == 1:
            end = size
        yield slice(start, end) if order is None else order[start:end]


def _scores(model, values, batch_size):
    model.eval()
    with torch.no_grad():
        return torch.cat([model.detector(values[idx]).squeeze(-1).cpu()
                          for idx in _batches(len(values), batch_size)]).numpy()


def _recalibrate(model, values, batch_size):
    model.enable_batch_norm_recalibration()
    with torch.no_grad():
        for idx in _batches(len(values), batch_size):
            model.detector(values[idx])
    model.eval()


def _selection_tpr(model, values, targets, cfg):
    from .rbc_metrics import calibrate_benign_threshold
    scores = _scores(model, values, cfg["batch_size"])
    threshold = calibrate_benign_threshold(scores[~targets], cfg["target_fpr"])["threshold"]
    return float(np.mean(scores[targets] > threshold))


def _train_detector(model, values, targets, selection, selection_targets, cfg, groups,
                    teacher=None):
    rng = np.random.default_rng(cfg["seed"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    best, stale, best_state = -float("inf"), 0, None
    history = []
    counts = torch.bincount(targets.long(), minlength=2).clamp_min(1)
    weights = len(targets) / (2 * counts.float())
    sample_weights = weights[targets.long()]
    benign_fraction = float((targets == 0).float().mean())
    for epoch in range(cfg["epochs"]):
        epoch_started = time.perf_counter()
        model.train()
        loss_sum, rows, single_class, max_streak, streak = 0., 0, 0, 0, 0
        for idx in _batches(len(values), cfg["batch_size"], rng):
            optimizer.zero_grad(set_to_none=True)
            batch_targets = targets[idx]
            output = model.detector(values[idx]).squeeze(-1)
            loss = normalized_detection_loss(
                output, batch_targets, sample_weight=sample_weights[idx],
                group_ids=groups[idx], worst_group_blend=cfg["worst_group_blend"],
                teacher_logits=None if teacher is None else teacher[idx],
                distillation_alpha=cfg["distillation_alpha"] if teacher is not None else 0.,
            )
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.))
            optimizer.step()
            clamp_binary_master_weights(model)
            loss_sum += float(loss.detach()) * len(idx)
            rows += len(idx)
            one = bool(batch_targets.min() == batch_targets.max())
            single_class += int(one)
            streak = streak + 1 if one else 0
            max_streak = max(max_streak, streak)
        from .rbc_metrics import calibrate_benign_threshold
        training_seconds = time.perf_counter() - epoch_started
        validation_started = time.perf_counter()
        selected_scores = _scores(model, selection, cfg["batch_size"])
        selected_threshold = calibrate_benign_threshold(
            selected_scores[~selection_targets], cfg["target_fpr"])["threshold"]
        score = float(np.mean(selected_scores[selection_targets] > selected_threshold))
        pr_auc = float(average_precision_score(selection_targets, selected_scores))
        history.append({"epoch": epoch, "loss": loss_sum / rows, "selection_pr_auc": pr_auc,
                        "selection_tpr_at_target_fpr": score,
                        "rows_consumed": rows, "resampled_rows": 0,
                        "single_class_batches": single_class, "max_single_class_streak": max_streak,
                        "last_gradient_norm": grad_norm,
                        "training_seconds": training_seconds,
                        "validation_seconds": time.perf_counter() - validation_started,
                        "benign_fraction": benign_fraction})
        if score > best:
            best, stale, best_state = score, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    return history


def _train_types(model, values, target, type_targets, *, epochs, batch_size, seed, lr,
                 cache_max_bytes=256 * 1024 * 1024):
    """Reuse the frozen binary representation without changing minibatches or updates."""
    if not model._detector_frozen or any(p.requires_grad for p in model.detector.parameters()):
        raise ValueError("type training requires a frozen detector")
    if epochs < 0 or batch_size < 2 or cache_max_bytes < 0:
        raise ValueError("invalid type training epochs, batch size, or cache budget")
    started = time.perf_counter()
    width = model.type_head.in_features
    required_bytes = len(values) * width  # final BNN activations are exactly -1 or +1
    hidden_cache = None
    if epochs > 1 and required_bytes <= cache_max_bytes:
        hidden_cache = torch.empty((len(values), width), dtype=torch.int8, device=values.device)
        with torch.no_grad():
            for idx in _batches(len(values), batch_size):
                hidden_cache[idx] = model.hidden_features(values[idx]).to(torch.int8)
    cache_seconds = time.perf_counter() - started
    optimizer = torch.optim.Adam(model.type_parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    attack_mask = target.bool()
    steps = 0
    for _ in range(epochs):
        model.train()
        for idx in _batches(len(values), batch_size, rng):
            batch_mask = attack_mask[idx]
            if not bool(batch_mask.any()):
                continue
            optimizer.zero_grad(set_to_none=True)
            if hidden_cache is None:
                with torch.no_grad():
                    hidden = model.hidden_features(values[idx])
            else:
                hidden = hidden_cache[idx].to(model.type_head.weight.dtype)
            logits = model.type_head(hidden)
            if model.attack_classes == 2:
                logits = logits.squeeze(-1)
            loss = conditional_type_loss(logits, type_targets[idx], attack_mask=batch_mask)
            loss.backward()
            optimizer.step()
            steps += 1
    return {"cached_hidden": hidden_cache is not None,
            "cache_bytes": required_bytes if hidden_cache is not None else 0,
            "cache_seconds": cache_seconds, "elapsed_seconds": time.perf_counter() - started,
            "optimizer_steps": steps, "epochs": epochs}


def run_rbc(config_path: str | Path) -> Path:
    from .rbc_encoder import RiskWeightedComparisonEncoder
    from .rbc_metrics import calibrate_benign_threshold, evaluate_rbc

    config = load_config(config_path)
    settings = config.get("rbc", {})
    seed = int(config["experiment"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(int(settings.get("threads", 1)))
    if config["dataset"].get("max_loaded_rows") is None:
        raise ValueError("RBC CSV runner requires an explicit dataset.max_loaded_rows memory limit")
    if config["dataset"].get("max_rows_per_file") or config["dataset"].get("max_rows_per_class"):
        raise ValueError("RBC partitions require unsampled source rows")
    if config["dataset"].get("storage") != "csv":
        raise ValueError("RBC runner uses normalized CSV sources; Parquet RBC training is not enabled")
    output = resolve_path(config, settings.get("output_dir", "runs/rbc"))
    if output.exists():
        raise FileExistsError(f"choose a fresh RBC output directory: {output}")
    started = time.perf_counter()
    loaded = load_dataset(config)
    parts = four_way_split(
        loaded.frame, strategy=settings.get("split_strategy", "temporal"),
        held_out_devices=settings.get("held_out_devices", []),
        held_out_attacks=settings.get("held_out_attacks", ["unknown_like"]),
        fractions=settings.get("split_fractions", [.60, .15, .10, .15]),
    )
    output.mkdir(parents=True, exist_ok=False)
    save_json(split_manifest(parts), output / "split_manifest.json")
    save_json(config, output / "resolved_config.json")
    save_json(loaded.provenance, output / "source_provenance.json")
    train = parts["train"]
    labels = sorted(set(train.behavior_label) - {"benign"})
    bits = int(settings.get("bit_budget", 64))
    encoder = RiskWeightedComparisonEncoder(bit_budget=bits, random_state=seed,
                                            raw_feature_cost_cap=settings.get("raw_feature_cost_cap"),
                                            max_swaps=int(settings.get("max_swaps", 8)))
    search_limit = int(settings.get("encoder_fit_rows", 10000))
    if search_limit < 8:
        raise ValueError("encoder_fit_rows must be at least eight")
    encoder_train = train if len(train) <= search_limit else train.sample(n=search_limit, random_state=seed)
    encoder.fit(encoder_train[loaded.feature_columns], encoder_train.behavior_label.ne("benign").to_numpy(),
                groups=encoder_train.device_id.astype(str).to_numpy()
                if encoder_train.device_id.nunique() > 1 else None,
                attack_groups=encoder_train.get("raw_attack", encoder_train.behavior_label).to_numpy())
    save_json({"encoder_fit_rows": len(encoder_train), "training_rows": len(train),
               "selection_scope": "training_only",
               "encoder_subsampled": len(encoder_train) < len(train),
               "detector_training_subsampled": False},
              output / "encoder_fit_budget.json")
    save_json(encoder.to_dict(), output / "encoder.json")
    values = {key: torch.from_numpy(encoder.transform(rows[loaded.feature_columns]))
              for key, rows in parts.items()}
    target = torch.tensor(train.behavior_label.ne("benign").to_numpy(), dtype=torch.float32)
    group_keys = train.device_id.astype(str) + ":" + train.get("raw_attack", train.behavior_label).astype(str)
    groups = torch.tensor(pd.factorize(group_keys, sort=True)[0])
    training = config["training"]
    cfg = {"seed": seed, "epochs": int(training["epochs"]),
           "batch_size": int(training["batch_size"]), "patience": int(training["patience"]),
           "learning_rate": float(training["learning_rate"]),
           "worst_group_blend": float(settings.get("worst_group_blend", 0.)),
           "target_fpr": float(settings.get("target_fpr", .001)),
           "distillation_alpha": float(settings.get("distillation_alpha", 0.))}
    if cfg["epochs"] < 1 or cfg["batch_size"] < 2 or cfg["patience"] < 1:
        raise ValueError("epochs/patience must be positive and batch_size at least two")
    model = DetectionFirstBNN(bits, list(config["model"]["hidden_dims"]), attack_classes=len(labels))
    teacher_logits = None
    if cfg["distillation_alpha"] > 0:
        teacher_started = time.perf_counter()
        # Raw-feature teacher: all statistics come exclusively from training data.
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        raw = train[loaded.feature_columns].replace([np.inf, -np.inf], np.nan)
        scaler = StandardScaler()
        imputer = SimpleImputer()
        rich = torch.tensor(scaler.fit_transform(imputer.fit_transform(raw)), dtype=torch.float32)
        teacher = nn.Sequential(nn.Linear(rich.shape[1], 128), nn.ReLU(), nn.Linear(128, 1))
        optimizer = torch.optim.Adam(teacher.parameters(), lr=cfg["learning_rate"])
        rng = np.random.default_rng(seed)
        for _ in range(cfg["epochs"]):
            for idx in _batches(len(rich), cfg["batch_size"], rng):
                optimizer.zero_grad(set_to_none=True)
                loss = normalized_detection_loss(teacher(rich[idx]).squeeze(-1), target[idx])
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            teacher_logits = teacher(rich).squeeze(-1).detach()
            selection_raw = parts["selection"][loaded.feature_columns].replace([np.inf, -np.inf], np.nan)
            selection_rich = torch.tensor(scaler.transform(imputer.transform(selection_raw)), dtype=torch.float32)
            teacher_scores = teacher(selection_rich).squeeze(-1).numpy()
        teacher_threshold = calibrate_benign_threshold(
            teacher_scores[parts["selection"].behavior_label.eq("benign")], cfg["target_fpr"])["threshold"]
        teacher_report = evaluate_rbc(
            teacher_scores, parts["selection"].behavior_label.to_numpy(), teacher_threshold,
            known_attack_labels=labels,
            unknown_attack_labels=sorted(set(parts["selection"].behavior_label) - set(labels) - {"benign"}),
            groups={"device": parts["selection"].device_id.to_numpy()})
        teacher_report["elapsed_seconds"] = time.perf_counter() - teacher_started
        teacher_report["scope"] = "selection_only; teacher excluded from deployment"
        save_json(teacher_report, output / "teacher_selection.json")
    history = _train_detector(
        model, values["train"], target, values["selection"],
        parts["selection"].behavior_label.ne("benign").to_numpy(), cfg, groups, teacher_logits)
    # Validate the collision-selected representation with an equally trained student.
    _recalibrate(model, values["train"], cfg["batch_size"])
    if encoder.replacement_history_:
        baseline_encoder = copy.deepcopy(encoder)
        baseline_encoder.comparisons_ = list(encoder.initial_comparisons_)
        baseline_encoder.replacement_history_ = []
        baseline_encoder.selection_collision_ = encoder.initial_selection_collision_
        baseline_values = {key: torch.from_numpy(baseline_encoder.transform(rows[loaded.feature_columns]))
                           for key, rows in parts.items()}
        torch.manual_seed(seed)
        baseline = DetectionFirstBNN(bits, list(config["model"]["hidden_dims"]), attack_classes=len(labels))
        selection_target = parts["selection"].behavior_label.ne("benign").to_numpy()
        baseline_history = _train_detector(baseline, baseline_values["train"], target,
                                           baseline_values["selection"], selection_target, cfg, groups, teacher_logits)
        _recalibrate(baseline, baseline_values["train"], cfg["batch_size"])
        candidate_tpr = _selection_tpr(model, values["selection"], selection_target, cfg)
        baseline_tpr = _selection_tpr(baseline, baseline_values["selection"], selection_target, cfg)
        save_json(encoder.to_dict(), output / "candidate_encoder.json")
        save_json({"candidate_selection_tpr": candidate_tpr, "baseline_selection_tpr": baseline_tpr,
                   "candidate_accepted": candidate_tpr >= baseline_tpr,
                   "selection_only": True, "baseline_history": baseline_history,
                   "additional_training_runs": 1}, output / "representation_selection.json")
        if candidate_tpr < baseline_tpr:
            model, encoder, values, history = baseline, baseline_encoder, baseline_values, baseline_history
        save_json(encoder.to_dict(), output / "encoder.json")
    model.freeze_detector()
    before = _scores(model, values["selection"], cfg["batch_size"])
    type_targets = torch.tensor([labels.index(label) if label in labels else 0
                                 for label in train.behavior_label])
    type_training = _train_types(
        model, values["train"], target, type_targets,
        epochs=int(settings.get("type_epochs", cfg["epochs"])), batch_size=cfg["batch_size"],
        seed=seed, lr=cfg["learning_rate"],
        cache_max_bytes=int(settings.get("type_cache_max_bytes", 256 * 1024 * 1024)))
    save_json(type_training, output / "type_training.json")
    if not np.array_equal(before, _scores(model, values["selection"], cfg["batch_size"])):
        raise RuntimeError("type training changed detection scores")
    model.eval()
    torch.save(model.state_dict(), output / "model.pt")
    save_json(history, output / "history.json")
    # Export is deliberately before calibration: quantization changes score ties.
    from .rbc_edge import export_rbc_edge, PackedRBCRuntime
    export_rbc_edge(model, output / "edge.npz", type_labels=labels)
    runtime = PackedRBCRuntime.load(output / "edge.npz")
    calibration_scores = runtime.predict(values["calibration"].numpy()).detection_scores[:, 0]
    calibration = calibrate_benign_threshold(
        calibration_scores[parts["calibration"].behavior_label.eq("benign")],
        float(settings.get("target_fpr", .001)), method=settings.get("calibration_method", "empirical"))
    save_json(calibration, output / "calibration.json")
    if calibration.get("threshold") is None:
        save_json({"status": "insufficient_calibration_support"}, output / "result.json")
        return output
    export_rbc_edge(model, output / "edge.npz", type_labels=labels,
                    detection_threshold=int(calibration["threshold"]))
    runtime = PackedRBCRuntime.load(output / "edge.npz")
    prediction = runtime.predict(values["test"].numpy())
    scores, type_indices = prediction.detection_scores[:, 0], prediction.attack_types
    types = np.asarray(labels)[type_indices]
    result = evaluate_rbc(scores, parts["test"].behavior_label.to_numpy(), calibration["threshold"],
                          known_attack_labels=labels, type_predictions=types,
                          unknown_attack_labels=sorted(set(parts["test"].behavior_label) - set(labels) - {"benign"}),
                          groups={"device": parts["test"].device_id.astype(str).to_numpy(),
                                  "attack": parts["test"].get("raw_attack", parts["test"].behavior_label).astype(str).to_numpy()})
    result["detector_preserved_during_type_training"] = True
    # Report measured Python-runtime latency separately from theoretical bit counts.
    benchmark_rows = min(len(parts["test"]), int(settings.get("benchmark_rows", 100)))
    latencies = []
    for index in range(benchmark_rows):
        tick = time.perf_counter_ns()
        runtime.predict(values["test"][index:index + 1].numpy())
        latencies.append((time.perf_counter_ns() - tick) / 1e6)
    result["packed_python_batch1_latency_ms"] = {
        "rows": benchmark_rows,
        "p50": float(np.percentile(latencies, 50)) if latencies else None,
        "p95": float(np.percentile(latencies, 95)) if latencies else None,
        "includes": "Python/Numpy packed runtime; excludes raw feature extraction and CSV loading",
        "machine": platform.machine(), "processor": platform.processor(),
    }
    result["elapsed_seconds"] = time.perf_counter() - started
    result["deployment_static_file_bytes"] = sum((output / name).stat().st_size
                                                 for name in ("edge.npz", "encoder.json", "calibration.json"))
    result["peak_ram_verified"] = False
    result["research_success"] = "not_assessed_without_matched_baseline_and_resource_measurement"
    save_json(result, output / "result.json")
    predictions = parts["test"][["row_uid", "device_id", "behavior_label"]].copy()
    predictions["attack_score"] = scores
    predictions["detected"] = scores > calibration["threshold"]
    predictions["type_prediction"] = types
    predictions.to_csv(output / "predictions.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    save_json({"seed": seed, "commit": commit.stdout.strip(), "python": platform.python_version(),
               "torch": torch.__version__, "device": "cpu", "threads": torch.get_num_threads(),
               "source_hashes": {str(path.relative_to(Path(__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in Path(__file__).parent.rglob("*.py")},
               "artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in output.iterdir() if p.is_file()}}, output / "manifest.json")
    return output
