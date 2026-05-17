"""

O pipeline MAS4RE reutiliza o ``PipelineState`` do Pydantic definido na
camada de domínio como seu esquema de estado LangGraph (ADR-005), evitando um
TypedDict paralelo e mantendo uma única fonte de verdade para o
estado de execução compartilhado.

"""

from domain.models import PipelineState

__all__ = ["PipelineState"]
