from enum import StrEnum


class Lang(StrEnum):
    """Idiomas suportados pelos prompts e datasets."""

    PT = "pt"
    EN = "en"
    ES = "es"


class InformationRegime(StrEnum):
    """What a run may use besides the requirement text and the dataset schema.

    ZERO_SHOT:          taxonomy and criteria only, no labelled examples and no trained models.
    HYBRID_EXPLORATORY: uses labelled data (retrieved examples or a trained verifier). Its
                        results must never be mixed with ZERO_SHOT ones.
    """

    ZERO_SHOT = "zero_shot"
    HYBRID_EXPLORATORY = "hybrid_exploratory"


class CoordinationProtocol(StrEnum):
    """How the two agents are allowed to interact (the experimental factor of the dissertation).

    FIXED:    classifier, then prioritizer, and no channel back (a one-way handoff).
    PEERS:    the prioritizer may ask the classifier to reconsider; the classifier agrees or
              refuses, and the prioritizer reconsiders once with the answer.
    MEDIATED: a mediator detects problems in the finished backlog and asks the prioritizer to
              reconsider, with evidence; the prioritizer agrees or refuses.
    """

    FIXED = "p0"
    PEERS = "p2"
    MEDIATED = "p3"


class RequirementType(StrEnum):
    FUNCTIONAL = "F"
    NON_FUNCTIONAL = "NF"


class NFRCategory(StrEnum):
    """Categorias NFR do dataset PROMISE NFR+ (Cleland-Huang et al., 2007)."""

    AVAILABILITY = "A"
    FAULT_TOLERANCE = "FT"
    LOOK_AND_FEEL = "LF"
    MAINTAINABILITY = "MN"
    OPERATIONAL = "O"
    PERFORMANCE = "PE"
    PORTABILITY = "PO"
    SCALABILITY = "SC"
    SECURITY = "SE"
    USABILITY = "US"
    FUNCTIONAL = "F"

    @classmethod
    def nfr_only(cls) -> list["NFRCategory"]:
        return [c for c in cls if c != cls.FUNCTIONAL]


class MoSCoWPriority(StrEnum):
    MUST_HAVE = "M"
    SHOULD_HAVE = "S"
    COULD_HAVE = "C"
    WONT_HAVE = "W"

    @property
    def score(self) -> float:
        return {"M": 1.0, "S": 0.75, "C": 0.5, "W": 0.25}[self.value]

    @property
    def ordinal(self) -> int:
        """Position on the 1 (Won't) .. 4 (Must) scale used by ordinal gold labels."""
        return {"W": 1, "C": 2, "S": 3, "M": 4}[self.value]

    @classmethod
    def from_ordinal(cls, ordinal: int) -> "MoSCoWPriority":
        by_ordinal = {priority.ordinal: priority for priority in cls}
        if ordinal not in by_ordinal:
            raise ValueError(f"MoSCoW ordinal must be one of {sorted(by_ordinal)}, got {ordinal}")
        return by_ordinal[ordinal]

    @property
    def label_pt(self) -> str:
        return {
            "M": "Deve ter",
            "S": "Deveria ter",
            "C": "Poderia ter",
            "W": "Não terá",
        }[self.value]
