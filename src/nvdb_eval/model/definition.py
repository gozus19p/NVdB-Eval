"""Definition data model for SensoComune dictionary entries."""

from dataclasses import dataclass
from typing import Dict, Any, Union


@dataclass
class Definition:
    """Represents a definition with its source.
    
    Attributes:
        glossa: The definition text
        source: The source of the definition (default: "tdm")
    """
    glossa: str
    source: str = "tdm"
    
    @classmethod
    def from_dict(cls, data: Union[str, Dict[str, Any]]) -> "Definition":
        """Create a Definition instance from a dictionary or string.
        
        Args:
            data: Either a string (simple glossa) or a dictionary containing
                  glossa and source fields
        
        Returns:
            A new Definition instance
        """
        if isinstance(data, str):
            return cls(glossa=data)
        
        return cls(
            glossa=data.get("glossa", ""),
            source=data.get("source", "tdm")
        )
