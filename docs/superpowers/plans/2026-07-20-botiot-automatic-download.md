# BoT-IoT Automatic Full-CSV Acquisition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing bootstrap acquire the complete BoT-IoT CSV archive automatically from a pinned public Kaggle revision while preserving explicit academic-license acknowledgement and the local-source override.

**Architecture:** Extend the immutable dataset registry with optional archive size and SHA-256 facts, then generalize the existing resumable downloader and acquisition orchestrator so any registered remote ZIP can use the same fail-closed path. The official UNSW page remains the dataset/license authority; provenance labels Kaggle as an approved public mirror, and downstream verification reuses the acquisition method recorded in the source manifest.

**Tech Stack:** Python 3.10-3.12 standard library, `unittest`, existing BitGuard bootstrap/downloader/extractor/manifest pipeline, JSON registry, Markdown documentation.

---

## File structure

- Modify `src/bitguard_bnn/bootstrap/types.py`: type and validate pinned archive metadata.
- Modify `src/bitguard_bnn/bootstrap/datasets.json`: pin the accepted Kaggle dataset revision.
- Modify `src/bitguard_bnn/bootstrap/registry.py`: enforce exact official identity and mirror transport facts.
- Modify `src/bitguard_bnn/bootstrap/cli.py`: make the local BoT-IoT source optional while retaining license gating.
- Modify `src/bitguard_bnn/bootstrap/download.py`: validate an optional pinned byte size in addition to SHA-256.
- Modify `src/bitguard_bnn/bootstrap/manifest.py`: represent the approved-mirror acquisition method without weakening manual or official-source rules.
- Modify `src/bitguard_bnn/bootstrap/orchestrator.py`: select automatic versus manual BoT-IoT acquisition, include it in disk preflight, and pass exact integrity facts to the downloader.
- Modify `src/bitguard_bnn/out_of_core/prepare.py`: rebuild raw-source provenance using the method already validated by the manifest.
- Modify `tests/test_bootstrap_cli.py`: lock registry and CLI behavior.
- Modify `tests/test_bootstrap_download.py`: lock pinned-size and approved-mirror manifest behavior.
- Modify `tests/test_bootstrap_integration.py`: lock automatic download, manual precedence, disk accounting, and downstream provenance.
- Modify `tests/test_out_of_core_prepare.py`: lock revalidation of automatic BoT-IoT provenance.
- Modify `tests/test_bootstrap_entry.py`: lock the fresh-machine wrapper command without a local source.
- Modify `README.md`: document the new command, source boundary, artifact sizes, fallback, and recovery behavior.

### Task 1: Pin the accepted BoT-IoT archive revision

**Files:**
- Modify: `tests/test_bootstrap_cli.py:30-126`
- Modify: `src/bitguard_bnn/bootstrap/types.py:8-82`
- Modify: `src/bitguard_bnn/bootstrap/datasets.json`
- Modify: `src/bitguard_bnn/bootstrap/registry.py:8-74`

- [ ] **Step 1: Write failing registry tests**

Replace the manual-only assertions with exact immutable transport facts and add malformed metadata cases:

```python
def test_registry_pins_botiot_public_mirror_revision(self):
    registry = load_registry()
    botiot = registry["botiot"]

    self.assertEqual(
        botiot.download_url,
        "https://www.kaggle.com/api/v1/datasets/download/"
        "vigneshvenkateswaran/bot-iot?datasetVersionNumber=1",
    )
    self.assertEqual(botiot.download_bytes, 1_257_092_644)
    self.assertEqual(
        botiot.download_sha256,
        "7869754e4b6192b45d4497be94cc34d621e1db81b6f76189e72ec4077e85bd75",
    )
    self.assertIn("research.unsw.edu.au", botiot.project_url)

def test_registry_rejects_invalid_archive_integrity_metadata(self):
    cases = (
        ("download_bytes", 0, "positive integer"),
        ("download_bytes", True, "positive integer"),
        ("download_sha256", "ABC", "lowercase SHA-256"),
        ("download_sha256", "0" * 63, "lowercase SHA-256"),
    )
    for field, value, message in cases:
        with self.subTest(field=field, value=value):
            payload = {
                name: spec.to_dict() for name, spec in load_registry().items()
            }
            payload["botiot"][field] = value
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "datasets.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_registry(path)
```

- [ ] **Step 2: Run the registry tests and verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_bootstrap_cli.BootstrapRegistryTest -v
```

Expected: failures because `DatasetSpec` has no `download_bytes` or
`download_sha256`, and BoT-IoT still forbids `download_url`.

- [ ] **Step 3: Add typed optional integrity fields**

Add validators and fields in `types.py`:

```python
def _optional_positive_integer(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer or null")
    return value


def _optional_sha256(value: object, field: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 or null")
    return value
```

Extend `DatasetSpec`, its exact field set, constructor, and `to_dict()` with:

```python
download_bytes: int | None
download_sha256: str | None
```

Require `download_url`, `download_bytes`, and `download_sha256` to be either all
present or all absent:

```python
download_values = (
    parsed.download_url,
    parsed.download_bytes,
    parsed.download_sha256,
)
if any(value is None for value in download_values) and any(
    value is not None for value in download_values
):
    raise ValueError(
        "download_url, download_bytes, and download_sha256 must be set together"
    )
```

- [ ] **Step 4: Pin the real archive metadata**

Set the BoT-IoT JSON entry to:

```json
{
  "name": "botiot",
  "project_url": "https://research.unsw.edu.au/projects/bot-iot-dataset",
  "download_url": "https://www.kaggle.com/api/v1/datasets/download/vigneshvenkateswaran/bot-iot?datasetVersionNumber=1",
  "download_bytes": 1257092644,
  "download_sha256": "7869754e4b6192b45d4497be94cc34d621e1db81b6f76189e72ec4077e85bd75",
  "doi": null,
  "license_name": "Academic research use",
  "expected_patterns": ["**/*.csv"],
  "required_columns": ["category", "subcategory", "saddr", "stime"]
}
```

Add `download_bytes` and `download_sha256` as `null` to N-BaIoT only if the
all-or-none rule is scoped to pinned downloads; retain its existing official
URL behavior by allowing a URL without integrity facts for the legacy official
source. Implement that distinction explicitly as:

```python
if parsed.download_url is None and (
    parsed.download_bytes is not None or parsed.download_sha256 is not None
):
    raise ValueError("download integrity metadata requires download_url")
if (parsed.download_bytes is None) != (parsed.download_sha256 is None):
    raise ValueError("download_bytes and download_sha256 must be set together")
```

- [ ] **Step 5: Enforce the exact mirror identity in the registry**

Add constants in `registry.py`:

```python
BOTIOT_DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "vigneshvenkateswaran/bot-iot?datasetVersionNumber=1"
)
BOTIOT_DOWNLOAD_BYTES = 1_257_092_644
BOTIOT_DOWNLOAD_SHA256 = (
    "7869754e4b6192b45d4497be94cc34d621e1db81b6f76189e72ec4077e85bd75"
)
```

Replace the manual-only validation with:

```python
if botiot.download_url != BOTIOT_DOWNLOAD_URL:
    raise ValueError("botiot.download_url must pin the approved Kaggle dataset version 1")
if botiot.download_bytes != BOTIOT_DOWNLOAD_BYTES:
    raise ValueError(f"botiot.download_bytes must be {BOTIOT_DOWNLOAD_BYTES}")
if botiot.download_sha256 != BOTIOT_DOWNLOAD_SHA256:
    raise ValueError("botiot.download_sha256 must match the accepted archive revision")
```

- [ ] **Step 6: Run tests and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_bootstrap_cli -v
```

Expected: all registry and existing CLI tests pass except the intentionally
unchanged manual-source requirement test, which Task 2 changes.

Commit only these four files with a Lore-formatted commit recording the pinned
Kaggle version, real archive size, SHA-256, and official UNSW authority.

### Task 2: Make automatic acquisition legal at the CLI boundary

**Files:**
- Modify: `tests/test_bootstrap_cli.py:128-234`
- Modify: `tests/test_bootstrap_entry.py:1-35`
- Modify: `src/bitguard_bnn/bootstrap/cli.py:24-74`

- [ ] **Step 1: Replace the source-required test with license-gated behavior**

Use these tests:

```python
def test_full_all_allows_automatic_botiot_with_license(self):
    options = parse_bootstrap_options(
        ["--full", "--accept-botiot-academic-license"]
    )

    self.assertEqual(options.datasets, ("nbaiot", "botiot"))
    self.assertIsNone(options.botiot_source)
    self.assertTrue(options.accepted_botiot_license)

def test_full_all_requires_botiot_license_before_acquisition(self):
    with self.assertRaisesRegex(ValueError, "accept-botiot-academic-license"):
        parse_bootstrap_options(["--full"])

def test_manual_botiot_source_remains_an_optional_override(self):
    options = parse_bootstrap_options(
        [
            "--dataset",
            "botiot",
            "--botiot-source",
            "official.zip",
            "--accept-botiot-academic-license",
        ]
    )

    self.assertEqual(options.botiot_source, Path("official.zip").resolve())
```

Add a wrapper test:

```python
def test_build_command_forwards_automatic_full_download(self):
    command = bootstrap.build_package_command(
        Path(".venv"),
        ["--full", "--accept-botiot-academic-license"],
    )

    self.assertEqual(
        command[-3:],
        ["bootstrap", "--full", "--accept-botiot-academic-license"],
    )
    self.assertNotIn("--botiot-source", command)
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_bootstrap_cli.BootstrapOptionsTest `
  tests.test_bootstrap_entry.BootstrapEntryTest -v
```

Expected: automatic full selection fails with the old
`BoT-IoT selection requires --botiot-source` error.

- [ ] **Step 3: Remove only the local-source requirement**

Delete:

```python
if "botiot" in datasets and args.botiot_source is None:
    raise ValueError("BoT-IoT selection requires --botiot-source from the official project")
```

Retain the license check unchanged and improve the option help:

```python
parser.add_argument(
    "--botiot-source",
    help="optional local BoT-IoT directory, ZIP, or RAR override",
)
parser.add_argument(
    "--accept-botiot-academic-license",
    action="store_true",
    help="confirm review of the official UNSW academic-use terms",
)
```

- [ ] **Step 4: Run tests and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_bootstrap_cli `
  tests.test_bootstrap_entry -v
```

Expected: all tests pass.

Commit the three files with a Lore message explaining that omission of the
source selects the pinned mirror, while the license acknowledgement remains
mandatory.

### Task 3: Enforce pinned archive size in the existing downloader

**Files:**
- Modify: `tests/test_bootstrap_download.py:215-470`
- Modify: `src/bitguard_bnn/bootstrap/download.py:91-102,619-829`

- [ ] **Step 1: Add failing byte-size tests**

Add:

```python
def test_expected_bytes_rejects_invalid_values_before_network_access(self):
    for value in (True, False, 0, -1, 1.5, "10"):
        with self.subTest(value=value), self.assertRaisesRegex(
            DownloadError, "expected_bytes"
        ):
            download_file(
                "https://example.test/archive",
                self.destination,
                expected_bytes=value,  # type: ignore[arg-type]
            )

def test_pinned_size_is_checked_before_publication(self):
    with _RangeServer() as server:
        with self.assertRaisesRegex(DownloadError, "expected_bytes|pinned size"):
            download_file(
                server.url,
                self.destination,
                expected_bytes=len(PAYLOAD) + 1,
                expected_sha256=PAYLOAD_SHA256,
            )

    self.assertFalse(self.destination.exists())
    self.assertTrue(self.partial.exists())

def test_existing_download_requires_both_pinned_size_and_hash(self):
    self.destination.write_bytes(PAYLOAD)

    with self.assertRaisesRegex(DownloadError, "expected_bytes|pinned size"):
        download_file(
            "https://example.test/archive",
            self.destination,
            expected_bytes=len(PAYLOAD) + 1,
            expected_sha256=PAYLOAD_SHA256,
        )
```

- [ ] **Step 2: Add a redirect-renewal resume test**

Extend `_RangeServer` with a `redirect-generation` mode that redirects
`/archive.zip` to `/redirected.zip?generation=N`, incrementing `N` on every
request to the stable archive URL. Let its first redirected response use the
existing short-body behavior, then switch the same server to normal range
behavior:

```python
def test_resume_renews_redirect_from_stable_source_url(self):
    half = len(PAYLOAD) // 2
    self.partial.write_bytes(PAYLOAD[:half])
    with _RangeServer(mode="redirect-generation") as server:
        result = download_file(
            server.url,
            self.destination,
            expected_bytes=len(PAYLOAD),
            expected_sha256=PAYLOAD_SHA256,
            chunk_size=97,
        )

    self.assertGreaterEqual(server.redirect_generation, 1)
    redirected_range = next(
        headers["Range"]
        for path, headers in zip(server.paths, server.requests)
        if path.startswith("/redirected.zip") and "Range" in headers
    )
    self.assertEqual(redirected_range, f"bytes={half}-")
    self.assertEqual(self.destination.read_bytes(), PAYLOAD)
    self.assertTrue(result.resumed)
```

The handler must append `self.path` to `fixture.paths`, increment
`fixture.redirect_generation` only for `/archive.zip`, return a relative
redirect containing that generation, and then serve `/redirected.zip` through
the existing range branch. This proves the downloader always resumes through
the stable registry URL and never persists a temporary redirect.

- [ ] **Step 3: Run tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_bootstrap_download.DownloadFileTest.test_expected_bytes_rejects_invalid_values_before_network_access `
  tests.test_bootstrap_download.DownloadFileTest.test_pinned_size_is_checked_before_publication `
  tests.test_bootstrap_download.DownloadFileTest.test_existing_download_requires_both_pinned_size_and_hash -v
```

Expected: `download_file()` rejects the unknown `expected_bytes` argument.

- [ ] **Step 4: Implement exact byte-size validation**

Add:

```python
def _normalize_expected_bytes(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DownloadError("expected_bytes must be a positive integer.")
    return value
```

Extend the signature:

```python
def download_file(
    source_url: str,
    destination: Path | str,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> DownloadResult:
```

Normalize `expected_bytes` at entry. In the existing-final branch, reject when
`size != expected_bytes` before returning reuse. In the verified-snapshot branch,
reject when `snapshot.byte_size != expected_bytes` before SHA comparison:

```python
if expected_bytes is not None and snapshot.byte_size != expected_bytes:
    raise DownloadError(
        "Downloaded content does not match expected_bytes; verified publication "
        f"was refused and {partial} remains resumable."
    )
```

- [ ] **Step 5: Run downloader tests and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_bootstrap_download.DownloadFileTest -v
```

Expected: all downloader tests pass.

Commit the two files with a Lore message explaining that the size pin augments
SHA-256 and does not replace it.

### Task 4: Record approved-mirror provenance

**Files:**
- Modify: `tests/test_bootstrap_download.py:963-1200`
- Modify: `src/bitguard_bnn/bootstrap/manifest.py:20-244`

- [ ] **Step 1: Replace the manual-only BoT-IoT manifest test**

Use:

```python
def test_botiot_allows_manual_or_exact_approved_mirror_provenance(self) -> None:
    spec = self.registry["botiot"]
    manual = build_source_manifest(
        self.root,
        spec,
        acquisition_method="manual-local-source",
    )
    mirrored = build_source_manifest(
        self.root,
        spec,
        acquisition_method="approved-public-mirror",
        acquisition_url=spec.download_url,
    )

    self.assertIsNone(manual.acquisition_url)
    self.assertEqual(mirrored.acquisition_url, spec.download_url)
    self.assertEqual(mirrored.acquisition_method, "approved-public-mirror")

    for method, url in (
        ("official-download", spec.download_url),
        ("approved-public-mirror", "https://storage.googleapis.com/signed?token=secret"),
        ("manual-local-source", spec.download_url),
    ):
        with self.subTest(method=method), self.assertRaisesRegex(
            SourceManifestError, "BoT-IoT|mirror|manual|URL"
        ):
            build_source_manifest(
                self.root,
                spec,
                acquisition_method=method,
                acquisition_url=url,
            )
```

- [ ] **Step 2: Run the test and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_bootstrap_download.SourceManifestTest.test_botiot_allows_manual_or_exact_approved_mirror_provenance -v
```

Expected: `approved-public-mirror` is not in the allowed method set.

- [ ] **Step 3: Add the method without weakening existing identities**

Set:

```python
ALLOWED_ACQUISITION_METHODS = frozenset(
    {"official-download", "manual-local-source", "approved-public-mirror"}
)
```

Replace the BoT-IoT validation branch with:

```python
if spec.name == "botiot":
    if method == "manual-local-source":
        if acquisition_url is not None:
            raise SourceManifestError(
                "Manual BoT-IoT acquisition must not record a URL."
            )
        return
    if method == "approved-public-mirror":
        if acquisition_url is None:
            raise SourceManifestError(
                "Approved BoT-IoT mirror acquisition requires its stable URL."
            )
        _validate_https_without_credentials(acquisition_url, "acquisition_url")
        if acquisition_url != spec.download_url:
            raise SourceManifestError(
                "BoT-IoT mirror URL does not match the pinned registry revision."
            )
        return
    raise SourceManifestError(
        "BoT-IoT must use manual-local-source or approved-public-mirror."
    )
```

- [ ] **Step 4: Run manifest tests and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_bootstrap_download.SourceManifestTest -v
```

Expected: all manifest tests pass.

Commit the two files with a Lore message that distinguishes transport
provenance from license authority.

### Task 5: Generalize orchestration to automatic BoT-IoT

**Files:**
- Modify: `tests/test_bootstrap_integration.py:32-110,690-930`
- Modify: `src/bitguard_bnn/bootstrap/orchestrator.py:39-170,2125-2265,2289-2490,2746-2770`

- [ ] **Step 1: Add an automatic-acquisition integration test**

Create a small ZIP fixture with BoT-IoT headers, select `botiot_source=None`,
and inject a downloader that records exact pins:

```python
def test_botiot_without_local_source_uses_pinned_mirror_and_provenance(self) -> None:
    mirror_zip = self.root / "mirror-botiot.zip"
    with zipfile.ZipFile(mirror_zip, "w") as archive:
        archive.writestr(
            "data_1.csv",
            "category,subcategory,saddr,stime,bytes,rate\n"
            "Normal,Normal,10.0.0.1,1.5,100,2.0\n"
            "DDoS,TCP,10.0.0.2,2.5,200,3.0\n",
        )
    observed: dict[str, object] = {}

    def mirror_download(url, destination, **kwargs):
        observed.update(url=url, **kwargs)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(mirror_zip, destination)
        payload = destination.read_bytes()
        return DownloadResult(
            destination=str(destination),
            byte_size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            resumed=False,
            restarted=False,
            reused=False,
            source_url=url,
            final_response_url="https://storage.googleapis.com/redacted",
        )

    options = replace(
        self.options,
        datasets=("botiot",),
        botiot_source=None,
    )
    report = run_bootstrap(
        options,
        dependencies=replace(
            self.dependencies(downloader=mirror_download),
            nbaiot_archive=None,
        ),
    )

    spec = load_registry()["botiot"]
    self.assertEqual(report["status"], "sources_verified")
    self.assertEqual(observed["url"], spec.download_url)
    self.assertEqual(observed["expected_bytes"], spec.download_bytes)
    self.assertEqual(observed["expected_sha256"], spec.download_sha256)
    manifest = json.loads(
        Path(report["manifests"]["botiot"]).read_text(encoding="utf-8")
    )
    self.assertEqual(manifest["acquisition_method"], "approved-public-mirror")
    self.assertEqual(manifest["acquisition_url"], spec.download_url)
```

Because the injected downloader intentionally uses a small fixture, it owns the
downloader contract and may return fixture metadata; the test asserts that
production pins were supplied to it rather than re-validating the fake bytes in
the orchestrator.

- [ ] **Step 2: Add manual precedence and license tests**

Add:

```python
def test_manual_botiot_source_takes_precedence_over_registered_mirror(self) -> None:
    downloader = Mock(side_effect=AssertionError("network must not be used"))

    report = run_bootstrap(
        replace(self.options, datasets=("botiot",)),
        dependencies=self.dependencies(downloader=downloader),
    )

    self.assertEqual(report["status"], "sources_verified")
    downloader.assert_not_called()
    manifest = json.loads(
        Path(report["manifests"]["botiot"]).read_text(encoding="utf-8")
    )
    self.assertEqual(manifest["acquisition_method"], "manual-local-source")
    self.assertIsNone(manifest["acquisition_url"])

def test_automatic_botiot_without_license_fails_before_downloader(self) -> None:
    downloader = Mock()
    report = run_bootstrap(
        replace(
            self.options,
            datasets=("botiot",),
            botiot_source=None,
            accepted_botiot_license=False,
        ),
        dependencies=self.dependencies(downloader=downloader),
    )

    self.assertEqual(report["failed_stage"], "preflight")
    self.assertIn("academic-license acknowledgement", str(report["error"]))
    downloader.assert_not_called()
```

- [ ] **Step 3: Run focused tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_bootstrap_integration.BootstrapAcquisitionIntegrationTest.test_botiot_without_local_source_uses_pinned_mirror_and_provenance `
  tests.test_bootstrap_integration.BootstrapAcquisitionIntegrationTest.test_manual_botiot_source_takes_precedence_over_registered_mirror `
  tests.test_bootstrap_integration.BootstrapAcquisitionIntegrationTest.test_automatic_botiot_without_license_fails_before_downloader -v
```

Expected: the first test fails because the orchestrator still requires a local
source; the latter tests lock unchanged safety behavior.

- [ ] **Step 4: Carry acquisition provenance in `SourceContext`**

Extend the typed dictionary:

```python
class SourceContext(TypedDict):
    kind: str
    digest: str | None
    bytes: int
    source: Path | None
    acquisition_method: str
    acquisition_url: str | None
```

Populate N-BaIoT with `official-download` and its registered URL. Populate a
manual BoT-IoT source with `manual-local-source` and `None`. When BoT-IoT has no
local source, require only the license acknowledgement and populate:

```python
spec = registry["botiot"]
assert spec.download_url is not None
assert spec.download_bytes is not None
assert spec.download_sha256 is not None
source_context["botiot"] = {
    "kind": "zip",
    "digest": spec.download_sha256,
    "bytes": spec.download_bytes,
    "source": None,
    "acquisition_method": "approved-public-mirror",
    "acquisition_url": spec.download_url,
}
```

- [ ] **Step 5: Include every remote archive in preflight**

Replace the N-BaIoT-only planned partial calculation with:

```python
planned_partial_bytes = sum(
    context["bytes"]
    for context in source_context.values()
    if context["source"] is None
)
```

Pass `planned_partial_bytes` to `estimate_resources()`. Keep the 12x archive
expansion estimate; for the pinned BoT-IoT archive it reserves more than the
observed 14,998,577,728 uncompressed bytes.

- [ ] **Step 6: Generalize the remote download branch**

Replace `if dataset == "nbaiot" and source is None` with `if source is None`.
Pass registry pins:

```python
result = deps.downloader(
    spec.download_url,
    candidate,
    expected_bytes=spec.download_bytes,
    expected_sha256=spec.download_sha256 or prior_hash,
)
dataset_report = result.to_dict()
dataset_report["method"] = context["acquisition_method"]
dataset_report["destination"] = str(destination)
```

For local fixtures and directories, keep copying behavior and set the report
method from `context["acquisition_method"]`.

- [ ] **Step 7: Build source manifests from the context**

Use:

```python
context = source_context[dataset]
manifest = build_source_manifest(
    raw_root,
    spec,
    acquisition_method=context["acquisition_method"],
    acquisition_url=context["acquisition_url"],
)
```

This ensures temporary signed redirect URLs never enter provenance.

- [ ] **Step 8: Run integration tests and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_bootstrap_integration -v
```

Expected: all acquisition integration tests pass.

Commit the two files with a Lore message recording manual precedence, pinned
automatic transport, and conservative disk accounting.

### Task 6: Preserve provenance through preparation verification

**Files:**
- Modify: `tests/test_out_of_core_prepare.py:80-105,580-675`
- Modify: `src/bitguard_bnn/out_of_core/prepare.py:705-729`

- [ ] **Step 1: Add an approved-mirror revalidation test**

Extend the test source-contract helper to accept explicit provenance:

```python
def _source_contract(
    dataset: str,
    raw_root: Path,
    root: Path,
    *,
    acquisition_method: str | None = None,
    acquisition_url: str | None = None,
):
    spec = load_registry()[dataset]
    method = acquisition_method or (
        "official-download" if dataset == "nbaiot" else "manual-local-source"
    )
    url = (
        acquisition_url
        if acquisition_method is not None
        else (spec.download_url if dataset == "nbaiot" else None)
    )
    manifest = build_source_manifest(
        raw_root,
        spec,
        acquisition_method=method,
        acquisition_url=url,
    )
```

Add a preparation test that passes
`acquisition_method="approved-public-mirror"` and
`acquisition_url=load_registry()["botiot"].download_url`, prepares the fixture,
and calls `verify_prepared_dataset()` successfully.

- [ ] **Step 2: Run the focused test and verify failure**

Run the new test by its exact unittest identifier after naming it
`test_approved_mirror_botiot_provenance_survives_reverification`:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_out_of_core_prepare.FullDatasetPreparationTests.test_approved_mirror_botiot_provenance_survives_reverification -v
```

Expected: failure because revalidation rebuilds every BoT-IoT manifest as
`manual-local-source`.

- [ ] **Step 3: Reuse already validated manifest metadata**

Replace the dataset-name heuristic in `_validate_source_contract_against_disk`:

```python
rebuilt = build_source_manifest(
    Path(prepared.raw_root),
    spec,
    acquisition_method=manifest.acquisition_method,
    acquisition_url=manifest.acquisition_url,
)
```

The manifest loader has already checked the method and URL against the current
registry, so this preserves provenance without trusting unvalidated descriptor
fields.

- [ ] **Step 4: Run preparation tests and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_out_of_core_prepare -v
```

Expected: all preparation tests pass.

Commit the two files with a Lore message explaining why provenance must be
replayed from the validated source manifest.

### Task 7: Document the one-command workflow and recovery boundary

**Files:**
- Modify: `README.md:63-174`

- [ ] **Step 1: Update prerequisites and commands**

Replace the mandatory manual-download prerequisite with:

```markdown
- Review the [official UNSW BoT-IoT project page](https://research.unsw.edu.au/projects/bot-iot-dataset)
  and use `--accept-botiot-academic-license` only when its academic-use terms
  apply. Automatic acquisition uses the pinned Kaggle dataset
  `vigneshvenkateswaran/bot-iot`, version 1, as a transport mirror; Kaggle is
  not the license authority.
- The full BoT-IoT archive is about 1.26 GB compressed and 15.0 GB extracted.
  Bootstrap verifies its pinned size and SHA-256 before extraction.
```

Use the new Windows command:

```powershell
.\bootstrap.ps1 --full --compute cu128 `
  --accept-botiot-academic-license `
  --data-root "$HOME\BitGuardData" `
  --runs-root "$HOME\BitGuardRuns"
```

Use the new Linux command:

```bash
./bootstrap.sh --full --compute auto \
  --accept-botiot-academic-license \
  --data-root "$HOME/bitguard-data" \
  --runs-root "$HOME/bitguard-runs"
```

Document the manual fallback:

```powershell
.\bootstrap.ps1 --full --compute cu128 `
  --botiot-source "$HOME\Datasets\BoT-IoT" `
  --accept-botiot-academic-license `
  --data-root "$HOME\BitGuardData" `
  --runs-root "$HOME\BitGuardRuns"
```

- [ ] **Step 2: Update the source and license boundary**

State that N-BaIoT uses official UCI transport, BoT-IoT uses the exact pinned
Kaggle mirror revision because current UNSW SharePoint links require Microsoft
authentication, and a local source always suppresses the mirror request.
Document that commercial use still requires agreement with the authors.

- [ ] **Step 3: Run documentation consistency searches**

Run:

```powershell
rg -n "never downloaded automatically|requires --botiot-source|Download the model-ready BoT-IoT" README.md src tests
```

Expected: no stale production documentation or error messages; historical
design/plan documents may retain their original statements.

- [ ] **Step 4: Commit documentation**

Commit `README.md` with a Lore message recording the authentication limitation,
mirror boundary, full archive sizes, and local fallback.

### Task 8: Full verification and remote metadata smoke check

**Files:**
- No source changes expected.

- [ ] **Step 1: Run formatting and static checks**

Run the repository's configured checks discovered from `pyproject.toml`. At
minimum:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
git diff --check
```

Expected: exit code 0.

- [ ] **Step 2: Run focused bootstrap and preparation suites**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_bootstrap_cli `
  tests.test_bootstrap_entry `
  tests.test_bootstrap_download `
  tests.test_bootstrap_preflight `
  tests.test_bootstrap_integration `
  tests.test_out_of_core_prepare -v
```

Expected: all tests pass.

- [ ] **Step 3: Run the complete repository test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected: all tests pass. If an unrelated dirty-worktree test fails, record its
exact pre-existing file and failure while still fixing every failure caused by
this change.

- [ ] **Step 4: Verify the real pinned archive evidence**

Confirm:

```powershell
(Get-Item C:\tmp\botiot-kaggle-v1.zip).Length
Get-FileHash -Algorithm SHA256 C:\tmp\botiot-kaggle-v1.zip
tar.exe -tf C:\tmp\botiot-kaggle-v1.zip
```

Expected:

```text
1257092644
7869754E4B6192B45D4497BE94CC34D621E1DB81B6F76189E72EC4077E85BD75
75 CSV entries ending with data_names.csv
```

- [ ] **Step 5: Run graph-assisted impact verification**

Update the code-review graph for
`C:\Users\js100\Desktop\coding\BitGuard`, request minimal context, detect
changes, and inspect impact radius, affected flows, and `tests_for` edges for
any medium/high-risk finding.

Expected: automatic acquisition affects only bootstrap acquisition,
provenance, preparation verification, wrapper tests, and documentation; every
reported production flow has focused test coverage.

- [ ] **Step 6: Review the staged diff and create the final Lore commit if needed**

Stage only the files listed in this plan. Verify unrelated user modifications
and untracked files remain unstaged. The final status report must include:

- exact changed files;
- reuse of the existing downloader instead of a Kaggle dependency;
- real archive size/SHA/ZIP evidence;
- focused and full test results;
- graph impact findings;
- remaining risk that Kaggle is a third-party transport and can become
  unavailable even though content drift fails closed.
