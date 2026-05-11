# ADR-008: Validação condicional de credenciais por modelo selecionado

- **Status:** Accepted
- **Data:** 2026-05-10
- **Autor:** Gabriela Zuppardo
- **SQ relacionada:** infra

## Contexto

`config/settings.py:13–14` define `anthropic_api_key: str = Field(default="")`
e idem para Groq. Falha tarde: o agente sobe, monta o prompt, chama o
LLM e só então recebe 401. Em dev local com Ollama, isso é inofensivo;
em produção, é frágil. O projeto roda local com Ollama (gratuito) e
sobe para Anthropic/Groq quando preciso.

## Decisão

- API keys são `SecretStr | None` em `Settings`.
- `config/validation.py::require_credentials(model, settings)`:
  - `ollama/*` → não exige chave.
  - `claude*` → exige `ANTHROPIC_API_KEY`; falha rápida se ausente.
  - `llama*|qwen*|mixtral*|gemma*` → exige `GROQ_API_KEY`.
- Chamada feita em `cli/commands/run.py` **antes** de instanciar o
  agente.
- `.env.example` versionado com todas as chaves documentadas.

## Consequências

**Positivas:**
- Dev local sem chaves: roda sem erro nem warning.
- Produção: erro claro e imediato com mensagem acionável.
- `SecretStr` evita logar chave por acidente.

**Negativas:**
- Validador precisa conhecer prefixos de modelo (acoplamento controlado).

**Neutras:**
- Segredos via env (`.env`, GitHub Secrets, SSM).

## Alternativas consideradas

- **Exigir todas as chaves sempre:** descartada — impede dev local.
- **Falhar só na chamada do LLM:** descartada — falha tarde e cara
  (após carregar dataset e construir prompt).

## Impacto na pesquisa

Permite rodar experimentos pequenos local (Ollama, gratuito) e
escalar para Anthropic/Groq sem mudar código — só `.env`.

## Referências
- `config/settings.py`, `config/validation.py` (a criar), `.env.example`
- ADRs relacionados: ADR-002