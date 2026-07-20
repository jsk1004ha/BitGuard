# BoT-IoT Automatic Full-CSV Acquisition Design

## Goal

Make a new machine able to acquire the complete model-ready BoT-IoT CSV
distribution, verify it, prepare it, and start training from the existing
bootstrap command without requiring a manual browser download.

The automatic path targets the complete CSV distribution rather than the
official 5% sample. The existing local `--botiot-source` path remains available
as a deterministic override and recovery path.

## Distribution and license boundary

The official UNSW project page remains the authoritative source for dataset
identity, academic-use terms, required citations, and commercial-use
restrictions:

<https://research.unsw.edu.au/projects/bot-iot-dataset>

The UNSW SharePoint folder currently requires an authenticated Microsoft
session, so it cannot support a fresh-machine, non-interactive bootstrap.
Automatic acquisition therefore uses the public Kaggle mirror
`vigneshvenkateswaran/bot-iot`, pinned to dataset version 1:

<https://www.kaggle.com/datasets/vigneshvenkateswaran/bot-iot>

The mirror is a transport provider, not the license authority. The bootstrap
must:

- continue to require `--accept-botiot-academic-license`;
- identify the UNSW project and its terms in reports and documentation;
- describe the acquisition method as a pinned public mirror, never as an
  official UNSW download;
- keep commercial use fail-closed in documentation by directing users to the
  authors for permission;
- never store Kaggle cookies, Microsoft credentials, or signed redirect URLs.

## Command behavior

The normal complete bootstrap becomes:

```powershell
.\bootstrap.ps1 --full --compute cu128 `
  --accept-botiot-academic-license `
  --data-root "$HOME\BitGuardData" `
  --runs-root "$HOME\BitGuardRuns"
```

When BoT-IoT is selected:

1. Absence of `--accept-botiot-academic-license` remains an error before any
   filesystem mutation.
2. If `--botiot-source` is present, the existing local directory/ZIP/RAR path
   takes precedence and no network request is made.
3. Otherwise, bootstrap downloads the pinned Kaggle version through the
   existing resumable HTTP acquisition layer.
4. Any identity, size, digest, archive-safety, PCAP, or CSV-schema mismatch
   stops the pipeline before publishing the raw dataset generation.
5. A successful acquisition continues through extraction, inspection,
   out-of-core preparation, and training exactly like the manual source path.

There is no automatic fallback to an unpinned alternate mirror. A failed public
download reports the official project URL and the local `--botiot-source`
recovery command.

## Registry and provenance

The dataset registry will retain the official UNSW project metadata while
adding a pinned acquisition descriptor for the Kaggle mirror. The descriptor
must include:

- provider and immutable dataset reference;
- dataset version;
- stable Kaggle API URL containing the explicit version;
- expected compressed archive size;
- expected SHA-256 digest captured from the accepted archive revision.

The source-manifest acquisition method will distinguish:

- `manual-local-source`;
- `official-download` for the existing UCI N-BaIoT source;
- `approved-public-mirror` for the pinned BoT-IoT transport.

The manifest records only the stable API URL, not the temporary signed Google
Storage redirect. Validation reloads the registry and rejects a provider,
version, size, digest, URL, method, or official project identity mismatch.

## Download, recovery, and storage safety

The current downloader is reused rather than adding a Kaggle SDK dependency.
The Kaggle public API endpoint redirects to a time-limited object URL and
supports byte ranges, so the downloader must preserve its partial file across
redirect renewal and resume using the stable API endpoint.

Before downloading, preflight includes:

- the compressed archive;
- the extracted full CSV tree;
- temporary extraction overhead;
- prepared shards and training workspace already estimated by the bootstrap.

Downloads are written only to bootstrap-owned temporary paths. Publication
remains atomic. A size or SHA-256 failure retires only bootstrap-owned partial
state and never touches a user-supplied source. Existing archive traversal,
symlink, nested-archive, and PCAP rejection remain in force.

## Testing

Behavior is locked with tests before implementation changes:

- CLI accepts automatic BoT-IoT acquisition only with the license
  acknowledgement and retains manual-source precedence.
- Registry validation pins the exact provider, dataset reference, version,
  stable URL, archive size, and SHA-256 digest.
- Manifest validation accepts `approved-public-mirror` only for the registered
  BoT-IoT revision and rejects signed redirect URLs or metadata drift.
- Downloader tests cover redirects, interruption, byte-range resume, a changed
  redirect on retry, incorrect size, and incorrect digest using local HTTP
  fixtures.
- Bootstrap integration tests cover automatic acquisition, manual override,
  resume, restart, failure cleanup, schema rejection, and successful handoff to
  preparation/training.
- Documentation and wrapper tests confirm that the new-machine command no
  longer requires `--botiot-source`.

The normal automated suite does not transfer the multi-gigabyte real archive.
A separately invoked remote smoke check validates the pinned metadata and a
small range response. The accepted archive SHA-256 is established once during
implementation from a complete download and then pinned in the registry.

## Alternative dataset decision

BoT-IoT remains the primary dataset because the current adapters, schema
inspection, preparation, training profiles, and recovery pipeline already
support its semantics.

TON_IoT is the preferred next dataset for external validation because it
contains newer heterogeneous IoT/IIoT telemetry, network, Linux, and Windows
sources. It is not a drop-in replacement: each source family has different
features, labels, and split semantics, so it requires a separate adapter and
configuration.

UNSW-NB15 is useful as a general network-intrusion baseline with an established
train/test partition. It is not IoT-specific and therefore should be added as a
comparison dataset rather than replacing BoT-IoT.

Adding either dataset is intentionally outside this change. Their future
implementation must preserve separate dataset identities instead of mapping
their schemas onto the BoT-IoT adapter.

## Failure behavior

Failures are explicit and recoverable:

- license not acknowledged: stop before mutation;
- mirror unavailable or rate-limited: retain resumable state when safe and
  print the rerun command;
- mirror identity changed: stop and require a reviewed registry update;
- insufficient disk: stop during preflight;
- unsupported archive contents or schema: quarantine bootstrap-owned output
  and do not prepare or train;
- manual override invalid: preserve the external source unchanged.

No failure silently switches datasets, downloads PCAP data, weakens integrity
checks, or falls back to a smaller sample.
