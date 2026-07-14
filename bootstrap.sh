#!/usr/bin/env sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

for candidate in python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
        exec "$candidate" "$SCRIPT_DIR/scripts/bootstrap.py" "$@"
    fi
done

echo "Python 3.10 through 3.12 is required." >&2
exit 1
