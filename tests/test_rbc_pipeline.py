import json

import yaml

from bitguard_bnn.demo import generate_demo
from bitguard_bnn.cli import main


def test_rbc_end_to_end(tmp_path):
    source = tmp_path / "data.csv"
    generate_demo(source, rows=1000)
    output = tmp_path / "run"
    config = {"dataset": {"type": "csv", "path": str(source), "max_loaded_rows": 2000},
              "model": {"hidden_dims": [8, 4]},
              "training": {"epochs": 1, "batch_size": 128, "patience": 1},
              "rbc": {"output_dir": str(output), "bit_budget": 8, "max_swaps": 1,
                      "type_epochs": 1, "distillation_alpha": .1, "worst_group_blend": .2}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["train-rbc", "--config", str(path)]) == 0
    report = json.loads((output / "result.json").read_text())
    assert report["detector_preserved_during_type_training"]
    assert (output / "manifest.json").exists()
    assert (output / "edge.npz").exists()
    assert report["peak_ram_verified"] is False
    predicted = tmp_path / "predicted.csv"
    assert main(["predict-rbc", "--run", str(output), "--input", str(source),
                 "--output", str(predicted)]) == 0
    import pandas as pd
    assert len(pd.read_csv(predicted)) == 1000
