# Running BitGuard on Modal

BitGuard can run its existing verified full-dataset bootstrap, preparation, training,
evaluation, and edge-export pipeline on Modal. The Modal integration does **not**
replace the scientific pipeline; it only supplies a GPU container and durable storage.

## What is persisted

The named Modal Volume `bitguard-bnn` is mounted at `/bitguard`, but BitGuard does **not** run directly on that filesystem. The bootstrap uses POSIX hard links and no-clobber rename operations that Modal Volume mounts do not provide. Instead, the runner restores a durable snapshot to a normal Linux working disk, runs BitGuard there, and mirrors it back periodically:

```text
/work/bitguard/                 # normal Linux working filesystem
├── BitGuardData/
└── BitGuardRuns/
        │
        │ rsync every 3 minutes + final sync
        ▼
/bitguard/snapshot-v2/          # persistent Modal Volume
├── BitGuardData/
└── BitGuardRuns/
```

The Function requests a 128 GiB ephemeral working disk. The repository and locked CUDA environment are baked into the Modal Image. BitGuard's normal atomic checkpoints and bootstrap state remain authoritative; the Volume is only the durable snapshot/restore layer.

## One-time setup

From a local clone of this repository:

```bash
python -m pip install "modal>=1.5,<1.6"
modal setup
```

Review the official UNSW BoT-IoT academic-use terms before enabling the BoT-IoT run.
The command-line acknowledgement is only a confirmation that you reviewed those
terms; it is not a license grant.

## Full N-BaIoT + BoT-IoT run

The cheapest default is T4:

```bash
modal run -n bitguard-bnn modal_train.py \
  --gpu T4 \
  --dataset all \
  --accept-botiot-license
```

For a faster GPU, change `--gpu` to `L4`, `A10`, `A100`, `L40S`, or `H100`.
BitGuard is small enough that data preparation and CPU-side Parquet preprocessing can
be a substantial part of wall-clock time, so a more expensive GPU is not guaranteed
to reduce total cost.

For a long run that should continue if the local terminal disconnects:

```bash
modal run -d -n bitguard-bnn modal_train.py \
  --gpu T4 \
  --dataset all \
  --accept-botiot-license
```

Modal Functions have a 24-hour maximum runtime in this integration. If the job is
interrupted or reaches that limit, run the **same command again**. The same Volume is
mounted and BitGuard reuses verified bootstrap stages and compatible training
checkpoints.

## Run one dataset only

N-BaIoT:

```bash
modal run -n bitguard-bnn modal_train.py --gpu T4 --dataset nbaiot
```

BoT-IoT:

```bash
modal run -n bitguard-bnn modal_train.py \
  --gpu T4 \
  --dataset botiot \
  --accept-botiot-license
```

## Inspect progress without allocating a GPU

```bash
modal run -n bitguard-status modal_train.py::read_status
```

You can also inspect the persistent filesystem directly:

```bash
modal volume ls bitguard-bnn /
modal volume ls bitguard-bnn snapshot-v2/BitGuardData/.bitguard
modal volume ls bitguard-bnn snapshot-v2/BitGuardRuns
```

## Download results

Download all run artifacts:

```bash
modal volume get bitguard-bnn snapshot-v2/BitGuardRuns ./BitGuardRuns
```

Or download a particular run directory after locating it with `modal volume ls`.

## Optional local BoT-IoT source

If automatic acquisition is unsuitable, upload an archive or directory to the Volume:

```bash
modal volume put bitguard-bnn ./bot-iot.zip uploads/bot-iot.zip
```

Then run:

```bash
modal run -n bitguard-bnn modal_train.py \
  --gpu T4 \
  --dataset botiot \
  --accept-botiot-license \
  --botiot-source uploads/bot-iot.zip
```

`--botiot-source` is intentionally restricted to paths inside the persistent Modal
Volume, so a retry sees exactly the same source bytes.

## Recovery and intentional restart

Normal interruption recovery does **not** need `--restart-stage`; rerun the same
command first. Use a stage restart only when BitGuard's bootstrap report explicitly
requires it or when you intentionally want to invalidate that stage and everything
after it:

```bash
modal run -n bitguard-bnn modal_train.py \
  --gpu T4 \
  --dataset all \
  --accept-botiot-license \
  --restart-stage train
```

Valid stage names are the same as the normal BitGuard bootstrap pipeline, such as
`environment`, `acquire`, `extract`, `inspect`, `shard`, `validate`, and `train`.

## Prepare-only

A prepare-only invocation is available:

```bash
modal run -n bitguard-bnn modal_train.py \
  --gpu T4 \
  --dataset all \
  --accept-botiot-license \
  --prepare-only
```

The current BitGuard bootstrap verifies the selected CUDA profile before executing the
pipeline, so this convenience path still allocates the selected GPU. For minimizing
cloud cost, the normal full run is generally preferable because it proceeds directly
from preparation into training in the same invocation.

## Cost notes

Modal charges the GPU, CPU, memory, and persistent Volume separately. The integration
requests 8 physical CPU cores and 32 GiB RAM because the full BitGuard path performs
substantial Parquet/Pandas preprocessing around a small BNN. T4 is therefore the
default: it leaves more of a free monthly compute allowance for CPU-side work and
retries. Use Modal's billing commands to inspect actual usage rather than estimating
from GPU price alone:

```bash
modal billing summary --for "this month"
modal billing report --for "this month" --show-resources
```


## Logs for detached runs

Name the ephemeral run with `-n bitguard-bnn` so it can be addressed by name while it is running:

```bash
modal app logs bitguard-bnn -f
```

If a run has already stopped, use the `ap-...` App ID printed by `modal run` and omit `-f`:

```bash
modal app logs ap-XXXXXXXX --tail 1000
```

`modal run` creates an ephemeral App. A stopped ephemeral App is not a currently deployed App, so following logs by its source-code `App(...)` name is not reliable unless the run was explicitly named with `-n`.
