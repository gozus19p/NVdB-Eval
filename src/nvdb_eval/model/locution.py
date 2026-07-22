"""Locution data model for multi-word expressions in SensoComune dictionary."""

from dataclasses import dataclass, field
from typing import List, Dict, Any

from .definition import Definition


@dataclass
class Locution:
    """Represents a locution (multi-word expression/polirematica).
    
    Attributes:
        expression: The locution text
        type: Type of locution (e.g., "idiomatic", "collocation")
        marks: Usage marks or labels
        definitions: List of definitions for this locution
        examples: Example sentences demonstrating this locution
    """
    expression: str = ""
    type: str = ""
    marks: List[str] = field(default_factory=list)
    definitions: List[Definition] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Locution":
        """Create a Locution instance from a dictionary.
        
        Args:
            data: Dictionary containing locution data
        
        Returns:
            A new Locution instance
        """
        return cls(
            expression=data.get("text", ""),
            type=data.get("type", ""),
            marks=data.get("marks", []),
            definitions=[
                Definition.from_dict(def_data)
                for def_data in data.get("definitions", [])
            ],
            examples=data.get("examples", [])
        )
    
    @property
    def meaning(self) -> str:
        """Compatibility property that returns the first definition.
        
        Returns:
            The first definition's glossa, or empty string if no definitions
        """
        return self.definitions[0].glossa if self.definitions else ""
    
    def get_all_meanings(self) -> List[str]:
        """Get all definition texts.
        
        Returns:
            List of all definition glosses
        """
        return [definition.glossa for definition in self.definitions]
