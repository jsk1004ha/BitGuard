# Running BitGuard on Modal

BitGuard can run its existing verified full-dataset bootstrap, preparation, training,
evaluation, and edge-export pipeline on Modal. The Modal integration does **not**
replace the scientific pipeline; it only supplies a GPU container and durable storage.

## What is persisted

The named Modal Volume `bitguard-bnn` is mounted at `/bitguard` and contains:

```text
/bitguard/
├── BitGuardData/
│   ├── .bitguard/        # bootstrap state/report, manifests, recovery metadata
│   ├── prepared/         # verified Parquet shards and preprocessing artifacts
│   └── ...               # acquired/extracted official dataset files
└── BitGuardRuns/
    ├── nbaiot_full/      # checkpoints, metrics, predictions, exports
    └── botiot_full/
```

The repository and locked CUDA environment are baked into the Modal Image. They do
not consume Volume space. A background thread commits the Volume every 60 seconds,
and the runner performs a final commit after the bootstrap process exits. BitGuard's
normal atomic checkpoints and bootstrap state remain authoritative.

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
modal run modal_train.py \
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
modal run -d modal_train.py \
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
modal run modal_train.py --gpu T4 --dataset nbaiot
```

BoT-IoT:

```bash
modal run modal_train.py \
  --gpu T4 \
  --dataset botiot \
  --accept-botiot-license
```

## Inspect progress without allocating a GPU

```bash
modal run modal_train.py::read_status
```

You can also inspect the persistent filesystem directly:

```bash
modal volume ls bitguard-bnn /
modal volume ls bitguard-bnn BitGuardData/.bitguard
modal volume ls bitguard-bnn BitGuardRuns
```

## Download results

Download all run artifacts:

```bash
modal volume get bitguard-bnn BitGuardRuns ./BitGuardRuns
```

Or download a particular run directory after locating it with `modal volume ls`.

## Optional local BoT-IoT source

If automatic acquisition is unsuitable, upload an archive or directory to the Volume:

```bash
modal volume put bitguard-bnn ./bot-iot.zip uploads/bot-iot.zip
```

Then run:

```bash
modal run modal_train.py \
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
modal run modal_train.py \
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
modal run modal_train.py \
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
