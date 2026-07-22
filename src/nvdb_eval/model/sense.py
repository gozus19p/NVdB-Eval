"""Sense data model for word meanings in SensoComune dictionary."""

from dataclasses import dataclass, field
from typing import List, Dict, Any

from .definition import Definition


@dataclass
class Sense:
    """Represents a sense/meaning of an acceptation.
    
    Attributes:
        number: Sense number/identifier
        marks: Usage marks or labels for this sense
        definitions: List of definitions for this sense
        examples: Example sentences demonstrating this sense
    """
    number: int
    marks: List[str] = field(default_factory=list)
    definitions: List[Definition] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sense":
        """Create a Sense instance from a dictionary.
        
        Handles both legacy format (single glossa field) and current format
        (definitions list).
        
        Args:
            data: Dictionary containing sense data
        
        Returns:
            A new Sense instance
        """
        definitions = []
        
        if "definitions" in data:
            # Current format with definitions list
            definitions = [
                Definition.from_dict(def_data)
                for def_data in data["definitions"]
            ]
        elif "glossa" in data and data["glossa"]:
            # Legacy format with single glossa - convert to Definition
            definitions.append(Definition(glossa=data["glossa"]))
        
        return cls(
            number=data.get("number", 0),
            marks=data.get("marks", []),
            definitions=definitions,
            examples=data.get("examples", [])
        )
    
    @property
    def glossa(self) -> str:
        """Compatibility property that returns the first definition.
        
        Returns:
            The first definition's glossa, or empty string if no definitions
        """
        return self.definitions[0].glossa if self.definitions else ""
    
    def get_all_glosse(self) -> List[str]:
        """Get all definition glosses.
        
        Returns:
            List of all definition texts
        """
        return [definition.glossa for definition in self.definitions]
