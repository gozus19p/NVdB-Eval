"""Lemma data model for dictionary entries in SensoComune."""

from dataclasses import dataclass, field
from typing import List, Dict, Any

from .acceptation import Acceptation
from .locution import Locution
from .sense import Sense


def objectId_from_dict(data: dict) -> str:
    return data["$oid"]


@dataclass
class Lemma:
    """Represents a lemma (dictionary entry) in SensoComune.
    
    Attributes:
        id: Unique identifier (MongoDB ObjectId)
        source: Source of the lemma data
        lemma: The lemma text
        grammar: Grammatical categories (e.g., ["s.m.", "s.f."])
        discr: Discriminator for homographs
        acceptations: List of acceptations (major sense groupings)
        locutions: List of multi-word expressions containing this lemma
    """
    id: str
    source: str
    lemma: str
    grammar: List[str] = field(default_factory=list)
    discr: int = 0
    acceptations: List[Acceptation] = field(default_factory=list)
    locutions: List[Locution] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Lemma":
        """Create a Lemma instance from a dictionary (JSON data).
        
        Args:
            data: Dictionary containing lemma data from JSON
        
        Returns:
            A new Lemma instance
        """
        return cls(
            id=objectId_from_dict(data["_id"]),
            source=data.get("source", ""),
            lemma=data.get("lemma", ""),
            grammar=data.get("grammar", []),
            discr=data.get("discr", 0),
            acceptations=[
                Acceptation.from_dict(acc) 
                for acc in data.get("acceptations", [])
            ],
            locutions=[
                Locution.from_dict(loc) 
                for loc in data.get("locutions", [])
            ]
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the lemma to a dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of the lemma
        """
        return {
            "_id": {"$oid": self.id},
            "source": self.source,
            "lemma": self.lemma,
            "grammar": self.grammar,
            "discr": self.discr,
            "acceptations": [
                {
                    "id": acc.id,
                    "usage": acc.usage,
                    "field": acc.domain_field,
                    "grammar": acc.grammar,
                    "is_link": acc.is_link,
                    "senses": [
                        {
                            "number": sense.number,
                            "marks": sense.marks,
                            "definitions": [
                                {
                                    "glossa": definition.glossa,
                                    "source": definition.source
                                }
                                for definition in sense.definitions
                            ],
                            "examples": sense.examples
                        }
                        for sense in acc.senses
                    ]
                }
                for acc in self.acceptations
            ],
            "locutions": [
                {
                    "text": loc.expression,
                    "type": loc.type,
                    "marks": loc.marks,
                    "definitions": [
                        {
                            "glossa": definition.glossa,
                            "source": definition.source
                        }
                        for definition in loc.definitions
                    ],
                    "examples": loc.examples
                }
                for loc in self.locutions
            ]
        }
    
    def get_all_examples(self) -> List[str]:
        """Get all example sentences from all senses and locutions.
        
        Returns:
            List of all example sentences
        """
        examples = []
        for acceptation in self.acceptations:
            for sense in acceptation.senses:
                examples.extend(sense.examples)
        for locution in self.locutions:
            examples.extend(locution.examples)
        return examples
    
    def get_senses_by_usage(self, usage: str) -> List[Sense]:
        """Get all senses for a given usage type (AU, CO, OB, etc.).
        
        Args:
            usage: Usage frequency indicator to filter by
        
        Returns:
            List of senses matching the usage type
        """
        senses = []
        for acceptation in self.acceptations:
            if acceptation.usage == usage:
                senses.extend(acceptation.senses)
        return senses
    
    def has_field(self, domain_field: str) -> bool:
        """Check if the lemma has acceptations in the specified domain.
        
        Args:
            domain_field: Domain/field to check for
        
        Returns:
            True if any acceptation belongs to the specified domain
        """
        return any(acc.domain_field == domain_field for acc in self.acceptations)
    
    def __str__(self) -> str:
        return f"Lemma('{self.lemma}', {len(self.acceptations)} accezioni)"
    
    def __repr__(self) -> str:
        return (
            f"Lemma(lemma='{self.lemma}', grammar={self.grammar}, "
            f"acceptations={len(self.acceptations)})"
        )
