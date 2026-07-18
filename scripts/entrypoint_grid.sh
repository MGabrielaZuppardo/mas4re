#!/usr/bin/env bash
set -euo pipefail

WORKERS="${GRID_WORKERS:-4}"
RESUME="${GRID_RESUME:-true}"
N="${GRID_N:-}"
BLOB_CONTAINER="${BLOB_CONTAINER:-experiments}"

CMD="python scripts/run_grid.py --workers $WORKERS"

if [[ "$RESUME" == "true" ]]; then
  CMD="$CMD --resume"
fi

if [[ -n "$N" ]]; then
  CMD="$CMD --n $N"
fi

echo "Iniciando: $CMD"
set +e
$CMD
EXIT_CODE=$?
set -e

if [[ -n "${AZURE_STORAGE_CONNECTION_STRING:-}" ]]; then
  echo "Enviando resultados para blob storage..."
  python - <<'PYEOF'
import os
import sys
from pathlib import Path
from azure.storage.blob import BlobServiceClient

conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
container = os.environ.get("BLOB_CONTAINER", "experiments")
client = BlobServiceClient.from_connection_string(conn_str)
container_client = client.get_container_client(container)

uploaded = 0
for directory in ["experiments/results", "experiments/traces"]:
    base = Path(directory)
    if not base.exists():
        continue
    for file in base.rglob("*"):
        if not file.is_file():
            continue
        blob_name = str(file).replace("\\", "/")
        with open(file, "rb") as f:
            container_client.upload_blob(blob_name, f, overwrite=True)
        uploaded += 1

print(f"  Upload concluído: {uploaded} arquivo(s)")
PYEOF
fi

exit $EXIT_CODE
