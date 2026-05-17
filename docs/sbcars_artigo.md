# MAS4RE → SBCARS 2026 — Construção do Artigo

> Documento-mestre da submissão. Consolida tema, estrutura, mapeamento
> código↔texto, cronograma e decisões estratégicas. Atualizar a cada ciclo.

---

## 1. Identificação

| Campo | Valor |
|---|---|
| Evento | SBCARS 2026 (20th Brazilian Symposium on Software Components, Architectures, and Reuse) @ CBSoft |
| Local / Data | São Paulo, 8–12/set/2026 |
| Trilha | Artigos de Pesquisa Técnica (10 + 2 páginas) |
| Tópico-âncora | Design, Arquitetura e Reutilização de Software Assistidos por IA |
| Título | *LLM Agents as Reusable Components: An Empirical Study* |
| Autores | Maria Gabriela Alves Zuppardo; Vinicius Cardoso Garcia (UFPE–CIn) |
| Idioma | Inglês |
| Submissão | JEMS3 + artefato Zenodo |
| Template | ACM-like CBSoft (Overleaf) |

### Datas-âncora
- Registro de resumo: **19/jun/2026**
- Submissão do artigo completo: **26/jun/2026**
- Notificação: 03/ago • Camera-ready: 10/ago • Simpósio: 09/set

---

## 2. Tema e contribuição central

**Tema:** trade-offs arquiteturais de **decomposição** em pipelines
assistidos por LLMs locais.

**Pitch (para o revisor SBCARS):** estudo empírico sobre como a
decomposição de um pipeline LLM em **múltiplos agentes especializados**
versus um **agente único** afeta qualidade, custo computacional e
modos de falha — validado num domínio real de classificação e
priorização de requisitos (PROMISE NFR+).

**O que NÃO vender:** "melhor classificador de NFR" (fit WER/SBES).
**O que vender:** decisão arquitetural assistida por IA + reúso de
componentes-agente + análise de falhas (fit SBCARS).

### Posicionamento vs. trabalhos próximos
- **NICE** (Rejithkumar & Anish, ICSE-SEIP 2025): fine-tuning de SLMs +
  explicabilidade. Citar como **paradigma complementar** (fine-tuning
  vs. nosso prompting; não competir em F1).
- **Zadenoori et al.** (arXiv 2025): SLM vs LLM em PROMISE. Citar como
  **justificativa empírica** para usar modelos locais.
- **AutoGen / MetaGPT / ChatDev**: frameworks multi-agente — citar em
  bloco como evidência do trend.

---

## 3. Perguntas de pesquisa

| RQ (artigo) | SQ (interno) | Pergunta | Métricas |
|---|---|---|---|
| RQ1 | SQ1 | Quão eficaz é cada agente especializado na sua tarefa? | Accuracy, Macro-F1, MCC |
| RQ2 | SQ2 | O multi-agente supera o agente único em qualidade e custo? | Wilcoxon pareado, Cliff's δ, tokens/latência |
| RQ3 | SQ3 | Quais padrões de falha emergem por arquitetura? | Contingência FailureMode × strategy, IC bootstrap |

> Convenção: **RQ** no artigo, **SQ** nos artefatos internos (ADRs,
> código). São a mesma coisa.

---

## 4. Estrutura do artigo (10 páginas ACM)

| Seção | Pág. | Status | Ciclo |
|---|---|---|---|
| §1 Introduction | 0,8 | ✅ v0 no Overleaf | C1 (v2 em S6) |
| §2 Background | 0,75 | ⏭️ | S5 |
| §3 Related Work | 1,0 | ⏭️ (refs coletadas) | S3→S5 |
| §4 MAS4RE: Architecture | 1,5 | ⏭️ (código pronto) | S5 |
| §5 Experimental Protocol | 1,0 | 🟡 §5.1+§5.2 redigidos (colar no Overleaf) | C1→S5 |
| §6 Results (RQ1/RQ2/RQ3) | 2,5 | ⏭️ | S4→S6 |
| §7 Discussion + Threats | 1,0 | ⏭️ | S6 |
| §8 Conclusion + Future Work | 0,25 | ⏭️ | S6 |
| Artifact Availability | — | ⏭️ obrigatório SBCARS | S5 |
| Referências | +2 | 🟡 15 entradas validadas | contínuo |

### §5 detalhado
- 5.1 Research Questions — redigido (PT/EN)
- 5.2 Dataset (PROMISE NFR+ PT/EN, 10 cat. NFR, tradução
  `deep_translator`) — redigido
- 5.3 Models (qwen2.5:7b, llama3.1:8b, phi3.5:3.8b; temp=0; seed=42) — S5
- 5.4 Metrics (qualidade + custo computacional + failure modes) — S5
- 5.5 Statistical analysis (Wilcoxon + Cliff's δ + bootstrap) — S5

---

## 5. Mapeamento código ↔ artigo

| Código (PR) | Seção do artigo |
|---|---|
| typed-core (PR #14): Lang, BaseAgent genérico | §4.2 (contrato reusável) |
| LangGraph mínima (PR #15): StateGraph | §4.4 (orquestração) + ADR-005 |
| Strategy + Runner (PR #16): manifest.json | §4.5 + §5 (reprodutibilidade) |
| Runner metrics (PR #17): results.json | §5.4 + §6.1/§6.2 |
| CLI Typer (PR #18): `mas4re run/eval/compare` | Artifact Availability |
| Failure taxonomy (PR #19): FailureMode/Record | §4.6 + §5.4 |
| Detectors (PR #20): DetectorChain v1 | §5.4 + §6.3 |
| Cross-agent check (PR #21): conflito inter-agente | §6.3 (achado próprio do pipeline) |
| TraceWriter (PR #22): trace.jsonl | §5 + Artifact Availability |
| Notebooks SQ1/2/3 (S4) | Tabelas/Figuras de §6 |

---

## 6. Progresso atual

### Código
- ✅ **S1**: typed-core + LangGraph mínima (PR #14, #15)
- ✅ **S2**: Strategy + Runner + métricas + CLI (PR #16, #17, #18)
- ✅ **E2E real validado**: `mas4re run --strategy pipeline --n 10`
  → accuracy 0.90, F1-macro 0.89, MCC 0.80 (preliminar, n=10)
- 🟡 **S3 em andamento**: PR #19 (taxonomia) ✅, PR #20 (detectores)
  ✅, PR #21 (cross-agent) em validação, PR #22 (trace) pendente

### Escrita
- ✅ §1 Introduction (v0) compilando no Overleaf
- 🟡 §5.1 + §5.2 redigidos — **pendente colar no Overleaf**
- ⏭️ demais seções: maratona de redação S5–S6

### Infra
- ✅ CI verde (ruff + format + mypy + pytest)
- ✅ pre-commit ativo (ruff 0.15.12 pinado = CI)
- ✅ `.gitattributes` LF; reprodutibilidade via manifest.json

---

## 7. Bibliografia (`mas4re.bib`)

15 entradas validadas. Slots:
- (a) NFR+LLM: ✅ NICE (`rejithkumar2025nice`)
- (b) Failure modes em LLM: ⏳ adiado para §3 (S3/S5)
- (c) SLM vs LLM em RE: ✅ Zadenoori (`zadenoori2025slm`)

Foundational (exceção 3 anos): Cleland-Huang 2007 (PROMISE).

---

## 8. Cronograma (code-first)

| Semana | Foco | Entrega |
|---|---|---|
| S1 ✅ | Fundação | typed-core + LangGraph + §1 |
| S2 ✅ | Runner/CLI | Strategy+Runner+métricas+CLI |
| S3 🟡 | SQ3 instrumentation | taxonomia+detectores+cross-check+trace |
| S4 | Grid + análise | 12 condições (3 mod × 2 idiomas × 2 arq) + notebooks SQ1/2/3 |
| S5 | Maratona texto 1 | §2,§3,§4,§5.3–5.5,§6.1 + Zenodo DOI |
| S6 | Maratona texto 2 | §6.2,§6.3,§7,§8,Abstract,§1 v2 + anonimização + **19/jun resumo JEMS** |
| S6.5 | Submissão | **26/jun JEMS3** |

⚠️ Decision point 07/jun (fim S4): se grid não fechar → pivot
**Artigo de Dados** (4 pág).

---

## 9. Decisões estratégicas (não reabrir sem motivo)

1. **Framing arquitetural obrigatório** — sem ele o paper cai fora do
   escopo SBCARS (não há tópico de RE puro no call).
2. **3 RQs = SQ1/SQ2/SQ3 do refactor** (eficácia / MAS vs baseline /
   padrões de falha). SQ3 é o diferencial.
3. **"computational cost"**, não "cost" — evita confusão monetária
   (rodamos local em Ollama).
4. **NICE e Zadenoori = complementares**, nunca competidores diretos.
5. **Contribuição 3 (reprodutibilidade/Zenodo)** entra no §1 P5 só em
   S5 (DOI só existe após publicar artefato).
6. **Idioma EN** — alcance + reaproveitamento (ICSME/SANER se rejeitar).
7. **Tradução PT automática** (`deep_translator`) → ameaça à validade
   declarada em §7; validar amostra n=50 se houver tempo (S4).

---

## 10. Riscos & Plano B

| Gatilho | Quando | Ação |
|---|---|---|
| Grid não fecha | fim S4 | reduzir a 2 modelos OU pivot Artigo de Dados |
| Sem sinal estatístico | fim S4 | reframe "trade-offs" (resultado negativo publica) |
| Inglês fraco | fim S6 | submeter + anotar ameaça à apresentação |
| §5 não no Overleaf | S2/S3 | colar (pendência aberta — baixo risco) |

---

## 11. Pendências imediatas

- [ ] Colar §5.1 + §5.2 no Overleaf (texto pronto)
- [ ] Fechar S3 (PR #21 merge + PR #22 TraceWriter)
- [ ] Report ao orientador (S1+S2 wrap-up; destacar E2E real F1=0.89)
- [ ] Buscar 1 ref slot (b) failure modes LLM para §3
