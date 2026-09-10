# Bootstrap and Dataset Acquisition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide one Windows/Linux command that creates a locked environment, downloads N-BaIoT, validates a user-supplied BoT-IoT source, safely extracts both, and resumes completed stages.

**Architecture:** Thin PowerShell/Bash wrappers call a standard-library Python entry point. The installed package owns an idempotent stage orchestrator, immutable source manifests, official dataset registry, resource preflight, resumable HTTP download, and safe archive extraction. This plan ends at verified extracted CSV sources; sharding and training are separate plans.

**Tech Stack:** Python 3.10-3.12, argparse, pathlib, urllib, hashlib, zipfile, subprocess, JSON, unittest, PyYAML, locked PyTorch/PyArrow profiles.

---

### Task 1: Lock profiles and wrapper contract

**Files:**
- Create: `bootstrap.ps1`
- Create: `bootstrap.sh`
- Create: `scripts/bootstrap.py`
- Create: `requirements/locks/full-base.txt`
- Create: `requirements/locks/torch-cpu.txt`
- Create: `requirements/locks/torch-cu118.txt`
- Create: `requirements/locks/torch-cu124.txt`
- Modify: `pyproject.toml`
- Create: `tests/test_bootstrap_entry.py`

- [ ] **Step 1: Write failing wrapper/argument tests**

Create tests that import `scripts/bootstrap.py` without project dependencies and verify argument forwarding:

```python
class BootstrapEntryTest(unittest.TestCase):
    def test_build_command_forwards_full_source_and_license(self):
        command = build_package_command(
            Path(".venv"),
            ["--full", "--botiot-source", "input.zip", "--accept-botiot-academic-license"],
        )
        self.assertEqual(command[-5:], [
            "bootstrap", "--full", "--botiot-source", "input.zip",
            "--accept-botiot-academic-license",
        ])

    def test_python_outside_supported_range_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Python 3.10 through 3.12"):
            validate_python_version((3, 13, 0))
```

- [ ] **Step 2: Run the entry tests and confirm RED**

Run: `python -m unittest tests.test_bootstrap_entry -v`

Expected: import failure because `scripts/bootstrap.py` does not exist.

- [ ] **Step 3: Add the standard-library environment entry point**

Implement these stable functions in `scripts/bootstrap.py`:

```python
SUPPORTED_PYTHON = ((3, 10), (3, 11), (3, 12))

def validate_python_version(version: tuple[int, int, int]) -> None:
    if version[:2] not in SUPPORTED_PYTHON:
        raise RuntimeError("bootstrap requires Python 3.10 through 3.12")

def venv_python(root: Path) -> Path:
    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    return root / relative

def build_package_command(environment: Path, forwarded: list[str]) -> list[str]:
    return [str(venv_python(environment)), "-m", "bitguard_bnn", "bootstrap", *forwarded]
```

The entry point must create `.venv`, install a repository-declared Torch profile before installing the editable package with `--no-deps`, install `full-base.txt`, and re-exec the package command. Select CPU, cu118, or cu124 from `--compute` and `nvidia-smi`; never silently downgrade a detected but broken CUDA profile.

Use known compatible locked profiles:

```text
# full-base.txt
numpy==2.2.3
pandas==2.2.3
scikit-learn==1.6.1
PyYAML==6.0.2
joblib==1.4.2
pyarrow==19.0.1
```

```text
# torch-cpu.txt
--index-url https://download.pytorch.org/whl/cpu
torch==2.6.0
```

The CUDA lock files use the same Torch version and official `cu118`/`cu124` indices.

- [ ] **Step 4: Add thin platform wrappers**

`bootstrap.ps1` resolves `py -3.12`, `py -3.11`, or `python`, then executes `scripts/bootstrap.py`. `bootstrap.sh` resolves `python3.12`, `python3.11`, or `python3.10`. Both pass all original arguments and return the Python exit code. They must not run package managers or extraction themselves.

- [ ] **Step 5: Register the full extra without changing base imports**

Add `full = ["pyarrow==19.0.1"]` to optional dependencies and keep PyArrow imports inside out-of-core modules. Do not import PyArrow when the normal demo CLI starts.

- [ ] **Step 6: Run entry tests and shell syntax checks**

Run:

```powershell
python -m unittest tests.test_bootstrap_entry -v
powershell -NoProfile -Command "[scriptblock]::Create((Get-Content -Raw bootstrap.ps1)) | Out-Null"
```

Run on Linux or CI: `bash -n bootstrap.sh`

Expected: all tests pass and both scripts parse.

- [ ] **Step 7: Commit only Task 1 files**

Use a Lore commit whose intent is environment reproducibility. Include `Tested:` with unit and shell syntax commands and `Constraint:` noting the Python prerequisite.

### Task 2: Bootstrap domain types, registry, and CLI

**Files:**
- Create: `src/bitguard_bnn/bootstrap/__init__.py`
- Create: `src/bitguard_bnn/bootstrap/types.py`
- Create: `src/bitguard_bnn/bootstrap/registry.py`
- Create: `src/bitguard_bnn/bootstrap/datasets.json`
- Create: `src/bitguard_bnn/bootstrap/cli.py`
- Modify: `src/bitguard_bnn/cli.py`
- Create: `tests/test_bootstrap_cli.py`

- [ ] **Step 1: Write failing parser and registry tests**

```python
def test_full_all_requires_botiot_source_and_license():
    with self.assertRaisesRegex(ValueError, "botiot-source"):
        parse_bootstrap_options(["--full", "--dataset", "all"])

def test_registry_contains_only_official_sources():
    registry = load_registry()
    self.assertEqual(registry["nbaiot"].doi, "10.24432/C5RC8J")
    self.assertIn("archive.ics.uci.edu", registry["nbaiot"].download_url)
    self.assertIsNone(registry["botiot"].download_url)
    self.assertIn("research.unsw.edu.au", registry["botiot"].project_url)
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `PYTHONPATH=src python -m unittest tests.test_bootstrap_cli -v`

Expected: missing bootstrap package.

- [ ] **Step 3: Define immutable types and registry loader**

Use frozen dataclasses with explicit serialization:

```python
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    project_url: str
    download_url: str | None
    doi: str | None
    license_name: str
    expected_patterns: tuple[str, ...]
    required_columns: tuple[str, ...]

@dataclass(frozen=True)
class BootstrapOptions:
    datasets: tuple[str, ...]
    botiot_source: Path | None
    data_root: Path
    runs_root: Path
    compute: str
    prepare_only: bool
    install_system_tools: bool
    accepted_botiot_license: bool
    restart_stage: str | None
```

The JSON registry pins the official UCI ZIP URL and metadata but stores no expected SHA until the repository has intentionally accepted a downloaded source revision. BoT-IoT has no automated URL.

- [ ] **Step 4: Add the package CLI parser**

Expose `bitguard bootstrap --full`. Validate source/license combinations before mutating the filesystem. Resolve all paths but retain user-facing originals in the report.

- [ ] **Step 5: Run parser/registry tests and existing CLI tests**

Run: `PYTHONPATH=src python -m unittest tests.test_bootstrap_cli tests.test_training_integration -v`

Expected: all pass; existing commands remain unchanged.

- [ ] **Step 6: Commit Task 2**

Use a Lore commit recording that official-source metadata is data, not embedded control flow.

### Task 3: Idempotent state and writer lock

**Files:**
- Create: `src/bitguard_bnn/bootstrap/state.py`
- Create: `tests/test_bootstrap_state.py`

- [ ] **Step 1: Write failing state transition tests**

```python
def test_stage_is_reused_only_when_input_signature_and_outputs_match():
    store = BootstrapStateStore(root / "state.json")
    output = root / "archive.zip"
    output.write_bytes(b"complete")
    store.complete("acquire", "input-a", [output])
    self.assertTrue(store.reusable("acquire", "input-a"))
    output.write_bytes(b"changed")
    self.assertFalse(store.reusable("acquire", "input-a"))

def test_restart_invalidates_stage_and_dependants():
    store.invalidate_from("extract", STAGE_ORDER)
    self.assertIn("acquire", store.completed_stages)
    self.assertNotIn("extract", store.completed_stages)
    self.assertNotIn("inspect", store.completed_stages)
```

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src python -m unittest tests.test_bootstrap_state -v`

- [ ] **Step 3: Implement atomic state and output fingerprints**

`BootstrapStateStore` writes `state.json.tmp` then `Path.replace`. Each output fingerprint contains resolved relative path, size, and SHA-256. `reusable` recomputes fingerprints. Store format version and reject unknown future versions.

- [ ] **Step 4: Implement an exclusive lock**

Create a lock file with `os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)`, PID, hostname, and start time. Refuse an active lock. Allow `--restart-stage` only after the stale PID is absent on the same host or after an explicit stale-lock recovery path; never delete a lock merely because it is old.

- [ ] **Step 5: Run state tests**

Run: `PYTHONPATH=src python -m unittest tests.test_bootstrap_state -v`

Expected: atomicity, reuse, invalidation, and lock tests pass.

- [ ] **Step 6: Commit Task 3**

Record atomic replacement and stale-lock policy in Lore trailers.

### Task 4: Resource and compute preflight

**Files:**
- Create: `src/bitguard_bnn/bootstrap/preflight.py`
- Create: `tests/test_bootstrap_preflight.py`

- [ ] **Step 1: Write failing resource tests**

```python
def test_insufficient_disk_fails_before_mutation():
    request = ResourceRequest(download=10, extracted=20, shards=15, temporary=5, reserve=10)
    with self.assertRaisesRegex(RuntimeError, "required=60.*available=59"):
        require_disk(request, available_bytes=59)

def test_detected_cuda_never_silently_falls_back():
    with self.assertRaisesRegex(RuntimeError, "CUDA profile verification failed"):
        choose_compute("auto", driver=DriverInfo(nvidia=True), torch_cuda=False)
```

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src python -m unittest tests.test_bootstrap_preflight -v`

- [ ] **Step 3: Implement resource math and platform checks**

Use `shutil.disk_usage`, `os.cpu_count`, and platform-safe RAM discovery (`GlobalMemoryStatusEx` on Windows, `os.sysconf` on Linux). Represent all byte estimates with integers. Inspect local archive sizes before extraction and include `.partial`, extracted, shard, evaluation, and reserve requirements.

- [ ] **Step 4: Implement compute verification**

Run `nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader`. After environment installation, import Torch, allocate a tensor on the selected device, synchronize, and record device/package versions. Explicit CPU remains valid even when a GPU exists.

- [ ] **Step 5: Run preflight tests**

Run: `PYTHONPATH=src python -m unittest tests.test_bootstrap_preflight -v`

- [ ] **Step 6: Commit Task 4**

Include the no-silent-fallback directive.

### Task 5: Resumable N-BaIoT download and immutable source manifest

**Files:**
- Create: `src/bitguard_bnn/bootstrap/download.py`
- Create: `src/bitguard_bnn/bootstrap/manifest.py`
- Create: `tests/test_bootstrap_download.py`

- [ ] **Step 1: Write a local range-server fixture and failing tests**

The fixture serves a byte string, honors `Range`, and records headers. Tests must pre-create half of `archive.zip.partial`, call `download_file`, assert `Range: bytes=<half>-`, final bytes, and SHA-256. Add a server-without-range test that restarts from byte zero rather than appending duplicate bytes.

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src python -m unittest tests.test_bootstrap_download -v`

- [ ] **Step 3: Implement streaming download**

Use `urllib.request`. Stream fixed chunks to `.partial`, call `flush` and `os.fsync`, validate `Content-Range` before append, compute SHA-256 over the completed file, then atomically rename. Never replace an already verified file.

- [ ] **Step 4: Implement source manifests**

Recursively hash verified source files in stable relative-path order. Store source project URL, DOI/license, size, hash, and acquisition method. BoT-IoT records `manual-local-source` and never copies a credential-bearing URL.

- [ ] **Step 5: Run download tests**

Run: `PYTHONPATH=src python -m unittest tests.test_bootstrap_download -v`

- [ ] **Step 6: Commit Task 5**

Record the HTTP restart behavior and immutable-source rule.

### Task 6: Safe extraction and schema inspection

**Files:**
- Create: `src/bitguard_bnn/bootstrap/extract.py`
- Create: `src/bitguard_bnn/bootstrap/inspect.py`
- Create: `tests/test_bootstrap_extract.py`
- Create: `tests/test_bootstrap_inspect.py`

- [ ] **Step 1: Write malicious archive tests**

Create ZIP fixtures containing `../escape.csv`, `/absolute.csv`, duplicate normalized names, and a symlink entry. Each must raise before writing outside the temporary extraction root. Mock a declared expanded size greater than available disk and assert preflight failure.

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src python -m unittest tests.test_bootstrap_extract tests.test_bootstrap_inspect -v`

- [ ] **Step 3: Implement safe ZIP extraction**

Resolve every destination under a temporary root and verify `destination.is_relative_to(root)`. Reject link file modes, duplicates, and unsupported encryption. Extract each member to a temporary file, fsync, then rename.

- [ ] **Step 4: Implement 7-Zip-compatible nested RAR extraction**

First list entries with `7z l -slt`, validate paths and declared sizes using the same policy, then extract into an isolated temporary directory. Never pass secrets or shell-composed strings; use argument arrays. When 7-Zip is absent, produce the exact supported `winget`, `apt`, or `dnf` command. Invoke it only when `--install-system-tools` is present.

- [ ] **Step 5: Inspect schemas in bounded chunks**

Read only headers and bounded CSV chunks. Verify required columns and numeric feature compatibility across files. Produce counts, class/device metadata, and rejected-row reasons. Default to failure when any row cannot be normalized.

- [ ] **Step 6: Run extraction and inspection tests**

Run: `PYTHONPATH=src python -m unittest tests.test_bootstrap_extract tests.test_bootstrap_inspect -v`

- [ ] **Step 7: Commit Task 6**

Record archive traversal protections and the explicit system-tool consent constraint.

### Task 7: Acquisition orchestrator and small-fixture integration

**Files:**
- Create: `src/bitguard_bnn/bootstrap/orchestrator.py`
- Create: `tests/test_bootstrap_integration.py`
- Modify: `README.md`
- Modify: `VALIDATION.md`

- [ ] **Step 1: Write an end-to-end acquisition test**

Use a local UCI ZIP fixture and local BoT-IoT directory. Run `preflight -> acquire -> extract -> inspect`, assert final manifests and state, then run again while mocks assert no HTTP request or extraction call. Modify one source byte and assert inspect plus downstream stages invalidate.

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src python -m unittest tests.test_bootstrap_integration -v`

- [ ] **Step 3: Implement the stage runner**

Represent stages as ordered records with `name`, `input_signature`, and `run` callable. Acquire the writer lock once around all mutable stages. Write a final JSON report on both success and failure, including the last completed stage and recovery command.

- [ ] **Step 4: Connect `bitguard bootstrap`**

The CLI calls the orchestrator and exits non-zero on any failed stage. `--prepare-only` stops after the later sharding stage added by Plan 2; until then it stops after inspection with status `sources_verified`.

- [ ] **Step 5: Document prerequisites and the one-time BoT-IoT step**

README must state that N-BaIoT is automatic, BoT-IoT is manually acquired from the official UNSW page, the academic license flag is an acknowledgement rather than a license grant, PCAP is excluded, and full CSV jobs are long-running.

- [ ] **Step 6: Run plan-level verification**

Run:

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.test_bootstrap_entry tests.test_bootstrap_cli tests.test_bootstrap_state tests.test_bootstrap_preflight tests.test_bootstrap_download tests.test_bootstrap_extract tests.test_bootstrap_inspect tests.test_bootstrap_integration -v
python -m compileall -q src scripts tests
git diff --check
```

Expected: all new tests pass; compile and diff checks are clean.

- [ ] **Step 7: Commit Task 7**

Use a Lore commit explaining why verified sources are the bootstrap foundation. State that full sharding/training follows in Plans 2 and 3.
