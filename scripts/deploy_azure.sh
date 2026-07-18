#!/usr/bin/env bash
# =============================================================================
# MAS4RE — Deploy no Azure Container Apps
#
# Uso:
#   bash scripts/deploy_azure.sh            # deploy completo
#   bash scripts/deploy_azure.sh --run      # deploy + dispara o grid job
#   bash scripts/deploy_azure.sh --job-only # só dispara o job (já deployado)
#
# Pré-requisitos:
#   - az CLI instalado e autenticado (az login)
#   - Docker instalado e rodando
#   - .env com AZURE_OPENAI_API_KEY e AZURE_OPENAI_ENDPOINT preenchidos
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuração — ajuste apenas aqui se necessário
# ---------------------------------------------------------------------------
RESOURCE_GROUP="mas4re-rg"
LOCATION="eastus"
REGISTRY_NAME="mas4reregistry"
ENVIRONMENT_NAME="mas4re-env"
JOB_NAME="mas4re-grid-job"
IMAGE_TAG="mas4re:latest"

# Lê credenciais do .env local
if [[ -f ".env" ]]; then
  export $(grep -v '^#' .env | grep -E 'AZURE_OPENAI|AZURE_STORAGE_CONNECTION_STRING|BLOB_CONTAINER' | xargs)
fi

AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-}"
AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-}"
CLASSIFIER_MODEL="${CLASSIFIER_MODEL:-azure/gpt-5-nano}"
PRIORITIZER_MODEL="${PRIORITIZER_MODEL:-azure/gpt-5-nano}"
GRID_WORKERS="${GRID_WORKERS:-4}"
GRID_N="${GRID_N:-}"  # vazio = dataset completo

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
RUN_JOB=false
JOB_ONLY=false
for arg in "$@"; do
  case $arg in
    --run)      RUN_JOB=true ;;
    --job-only) JOB_ONLY=true; RUN_JOB=true ;;
  esac
done

# ---------------------------------------------------------------------------
# Validações
# ---------------------------------------------------------------------------
if [[ -z "$AZURE_OPENAI_API_KEY" ]]; then
  echo "ERRO: AZURE_OPENAI_API_KEY não definida no .env"
  exit 1
fi
if [[ -z "$AZURE_OPENAI_ENDPOINT" ]]; then
  echo "ERRO: AZURE_OPENAI_ENDPOINT não definida no .env"
  exit 1
fi

echo ""
echo "=== MAS4RE Azure Deploy ==="
echo "  Resource Group : $RESOURCE_GROUP"
echo "  Location       : $LOCATION"
echo "  Registry       : $REGISTRY_NAME"
echo "  Job            : $JOB_NAME"
echo "  Model          : $CLASSIFIER_MODEL"
echo "  Workers        : $GRID_WORKERS"
echo "  N samples      : ${GRID_N:-full}"
echo ""

if $JOB_ONLY; then
  echo "[skip] Build e deploy — disparando job existente..."
else
  # -------------------------------------------------------------------------
  # 1. Resource group
  # -------------------------------------------------------------------------
  echo "[1/5] Verificando resource group..."
  az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none

  # -------------------------------------------------------------------------
  # 2. Container Registry
  # -------------------------------------------------------------------------
  echo "[2/5] Criando Container Registry..."
  az acr create \
    --name "$REGISTRY_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --sku Basic \
    --admin-enabled true \
    --output none 2>/dev/null || echo "  (registry já existe)"

  # -------------------------------------------------------------------------
  # 3. Build local e push da imagem
  # -------------------------------------------------------------------------
  echo "[3/5] Build local e push da imagem..."
  REGISTRY_SERVER="${REGISTRY_NAME}.azurecr.io"

  az acr login --name "$REGISTRY_NAME"

  docker build -t "${REGISTRY_SERVER}/${IMAGE_TAG}" -f Dockerfile .
  docker push "${REGISTRY_SERVER}/${IMAGE_TAG}"

  # -------------------------------------------------------------------------
  # 4. Container Apps Environment
  # -------------------------------------------------------------------------
  echo "[4/5] Criando Container Apps Environment..."
  az containerapp env create \
    --name "$ENVIRONMENT_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location brazilsouth \
    --logs-destination none \
    --output none 2>/dev/null || echo "  (environment já existe)"


  # -------------------------------------------------------------------------
  # 5. Container Apps Job
  # -------------------------------------------------------------------------
  echo "[5/5] Criando Container Apps Job..."

  GRID_RESUME="true"

  REGISTRY_PASS=$(az acr credential show \
    --name "$REGISTRY_NAME" \
    --query "passwords[0].value" -o tsv)

  # Habilita acesso do Container Apps ao ACR via role assignment
  ACR_ID=$(az acr show --name "$REGISTRY_NAME" --query id -o tsv)
  ENV_IDENTITY=$(az containerapp env show \
    --name "$ENVIRONMENT_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query "identity.principalId" -o tsv 2>/dev/null || echo "")

  JOB_EXISTS=$(az containerapp job show \
    --name "$JOB_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query "name" -o tsv 2>/dev/null || echo "")

  JOB_CREATE_LOG=$(mktemp)
  chmod 600 "$JOB_CREATE_LOG"
  trap 'rm -f "$JOB_CREATE_LOG"' EXIT

  # Segredos (API key, connection string) são gravados via --secrets e
  # referenciados nas env-vars com secretref: — o Azure CLI nunca ecoa o
  # valor de volta no stdout/stderr, ao contrário de passá-los diretamente
  # em --env-vars.
  if [[ -z "$JOB_EXISTS" ]]; then
    az containerapp job create \
      --name "$JOB_NAME" \
      --resource-group "$RESOURCE_GROUP" \
      --environment "$ENVIRONMENT_NAME" \
      --trigger-type Manual \
      --replica-timeout 7200 \
      --replica-retry-limit 1 \
      --replica-completion-count 1 \
      --parallelism 1 \
      --image "${REGISTRY_SERVER}/${IMAGE_TAG}" \
      --registry-server "$REGISTRY_SERVER" \
      --registry-username "$REGISTRY_NAME" \
      --registry-password "$REGISTRY_PASS" \
      --cpu 2 \
      --memory 4Gi \
      --secrets \
        "azure-openai-api-key=$AZURE_OPENAI_API_KEY" \
        "azure-storage-conn=${AZURE_STORAGE_CONNECTION_STRING:-}" \
      --env-vars \
        "AZURE_OPENAI_API_KEY=secretref:azure-openai-api-key" \
        "AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT" \
        "CLASSIFIER_MODEL=$CLASSIFIER_MODEL" \
        "PRIORITIZER_MODEL=$PRIORITIZER_MODEL" \
        "GRID_WORKERS=$GRID_WORKERS" \
        "GRID_RESUME=$GRID_RESUME" \
        "GRID_N=$GRID_N" \
        "AZURE_STORAGE_CONNECTION_STRING=secretref:azure-storage-conn" \
        "BLOB_CONTAINER=${BLOB_CONTAINER:-experiments}" \
      --command "bash" \
      --args "scripts/entrypoint_grid.sh" \
      2>&1 | tee "$JOB_CREATE_LOG"
    # Se falhou por auth, tenta com admin explícito
    if grep -q "UNAUTHORIZED\|authentication" "$JOB_CREATE_LOG" 2>/dev/null; then
      echo "  Tentando com credenciais admin explícitas..."
      REGISTRY_PASS2=$(az acr credential show --name "$REGISTRY_NAME" --query "passwords[1].value" -o tsv)
      az containerapp job create \
        --name "$JOB_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --environment "$ENVIRONMENT_NAME" \
        --trigger-type Manual \
        --replica-timeout 7200 \
        --replica-retry-limit 1 \
        --replica-completion-count 1 \
        --parallelism 1 \
        --image "${REGISTRY_SERVER}/${IMAGE_TAG}" \
        --registry-server "$REGISTRY_SERVER" \
        --registry-username "$REGISTRY_NAME" \
        --registry-password "$REGISTRY_PASS2" \
        --cpu 2 \
        --memory 4Gi \
        --secrets \
          "azure-openai-api-key=$AZURE_OPENAI_API_KEY" \
          "azure-storage-conn=${AZURE_STORAGE_CONNECTION_STRING:-}" \
        --env-vars \
          "AZURE_OPENAI_API_KEY=secretref:azure-openai-api-key" \
          "AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT" \
          "CLASSIFIER_MODEL=$CLASSIFIER_MODEL" \
          "PRIORITIZER_MODEL=$PRIORITIZER_MODEL" \
          "GRID_WORKERS=$GRID_WORKERS" \
          "GRID_RESUME=$GRID_RESUME" \
          "GRID_N=$GRID_N" \
          "AZURE_STORAGE_CONNECTION_STRING=secretref:azure-storage-conn" \
          "BLOB_CONTAINER=${BLOB_CONTAINER:-experiments}" \
        --command "bash" \
        --args "scripts/entrypoint_grid.sh"
    fi
  else
    echo "  (job já existe — atualizando imagem...)"
    az containerapp job update \
      --name "$JOB_NAME" \
      --resource-group "$RESOURCE_GROUP" \
      --image "${REGISTRY_SERVER}/${IMAGE_TAG}" \
      --output none
  fi

  echo ""
  echo "Deploy concluído."
fi

# ---------------------------------------------------------------------------
# Disparar o job
# ---------------------------------------------------------------------------
if $RUN_JOB; then
  echo ""
  echo "Disparando grid job..."
  EXECUTION=$(az containerapp job start \
    --name "$JOB_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query "name" -o tsv)

  echo "Job iniciado: $EXECUTION"
  echo ""
  echo "Acompanhe os logs:"
  echo "  az containerapp job execution show \\"
  echo "    --name $JOB_NAME \\"
  echo "    --resource-group $RESOURCE_GROUP \\"
  echo "    --job-execution-name $EXECUTION"
  echo ""
  echo "  az containerapp job logs show \\"
  echo "    --name $JOB_NAME \\"
  echo "    --resource-group $RESOURCE_GROUP \\"
  echo "    --execution $EXECUTION --follow"
  echo ""
  echo "Quando o job terminar, baixe os resultados:"
  echo "  az storage blob download-batch \\"
  echo "    --account-name mas4re \\"
  echo "    --source experiments \\"
  echo "    --destination ."
fi

echo ""
echo "Pronto!"
