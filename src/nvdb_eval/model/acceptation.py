"""Acceptation data model for word senses in SensoComune dictionary."""

from dataclasses import dataclass, field
from typing import List, Dict, Any

from .sense import Sense


@dataclass
class Acceptation:
    """Represents an acceptation (word sense grouping) of a lemma.
    
    Attributes:
        id: Unique identifier for the acceptation
        usage: Usage frequency indicator (AU, CO, OB, etc.)
        domain_field: Domain or field of usage
        grammar: Grammatical information
        is_link: Whether this is a reference/link to another entry
        senses: List of senses within this acceptation
    """
    id: str
    usage: str = ""
    domain_field: str = ""
    grammar: List[str] = field(default_factory=list)
    is_link: bool = False
    senses: List[Sense] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Acceptation":
        """Create an Acceptation instance from a dictionary.
        
        Args:
            data: Dictionary containing acceptation data
        
        Returns:
            A new Acceptation instance
        """
        return cls(
            id=data.get("id", ""),
            usage=data.get("usage", ""),
            domain_field=data.get("field", ""),
            grammar=data.get("grammar", []),
            is_link=data.get("is_link", False),
            senses=[Sense.from_dict(sense) for sense in data.get("senses", [])]
        )
