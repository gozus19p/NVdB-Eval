"""SensoComune collection data model for managing dictionary entries."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterator

from .lemma import Lemma


@dataclass
class SensoComune:
    """Represents the entire SensoComune lemma collection.
    
    Attributes:
        lemmas: List of all lemmas in the collection
    """
    lemmas: List[Lemma] = field(default_factory=list)
    
    @classmethod
    def from_json_list(cls, data: List[Dict[str, Any]]) -> "SensoComune":
        """Create a SensoComune instance from a list of JSON dictionaries.
        
        Args:
            data: List of dictionaries containing lemma data
        
        Returns:
            A new SensoComune instance with all loaded lemmas
        """
        return cls(lemmas=[Lemma.from_dict(item) for item in data])
    
    def get_lemma_by_text(self, text: str) -> Optional[Lemma]:
        """Find a lemma by its text (case-insensitive).
        
        Args:
            text: The lemma text to search for
        
        Returns:
            The first matching Lemma, or None if not found
        """
        text_lower = text.lower()
        for lemma in self.lemmas:
            if lemma.lemma.lower() == text_lower:
                return lemma
        return None
    
    def get_lemmas_by_grammar(self, grammar: str) -> List[Lemma]:
        """Get all lemmas of a specific grammatical category.
        
        Args:
            grammar: Grammatical category to filter by (e.g., "s.m.", "v.tr.")
        
        Returns:
            List of lemmas matching the grammatical category
        """
        return [lemma for lemma in self.lemmas if grammar in lemma.grammar]
    
    def get_lemmas_by_source(self, source: str) -> List[Lemma]:
        """Get all lemmas from a specific source.
        
        Args:
            source: Source identifier to filter by
        
        Returns:
            List of lemmas from the specified source
        """
        return [lemma for lemma in self.lemmas if lemma.source == source]
    
    def search_in_glosses(self, query: str) -> List[Lemma]:
        """Search for lemmas containing the query in their definitions.
        
        Args:
            query: Search term (case-insensitive)
        
        Returns:
            List of lemmas whose definitions contain the query
        """
        query_lower = query.lower()
        results = []
        
        for lemma in self.lemmas:
            found = False
            for acceptation in lemma.acceptations:
                if found:
                    break
                for sense in acceptation.senses:
                    if found:
                        break
                    for definition in sense.definitions:
                        if query_lower in definition.glossa.lower():
                            results.append(lemma)
                            found = True
                            break
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate collection statistics.
        
        Returns:
            Dictionary containing various statistics about the collection
        """
        total_lemmas = len(self.lemmas)
        total_acceptations = sum(len(lemma.acceptations) for lemma in self.lemmas)
        total_senses = sum(
            len(acc.senses) 
            for lemma in self.lemmas 
            for acc in lemma.acceptations
        )
        
        grammar_counts: Dict[str, int] = {}
        usage_counts: Dict[str, int] = {}
        
        for lemma in self.lemmas:
            for gram in lemma.grammar:
                grammar_counts[gram] = grammar_counts.get(gram, 0) + 1
            for acc in lemma.acceptations:
                if acc.usage:
                    usage_counts[acc.usage] = usage_counts.get(acc.usage, 0) + 1
        
        return {
            "total_lemmas": total_lemmas,
            "total_acceptations": total_acceptations,
            "total_senses": total_senses,
            "grammar_distribution": grammar_counts,
            "usage_distribution": usage_counts,
            "avg_acceptations_per_lemma": (
                total_acceptations / total_lemmas if total_lemmas > 0 else 0
            ),
            "avg_senses_per_lemma": (
                total_senses / total_lemmas if total_lemmas > 0 else 0
            )
        }
    
    def __len__(self) -> int:
        return len(self.lemmas)
    
    def __iter__(self) -> Iterator[Lemma]:
        return iter(self.lemmas)
    
    def __str__(self) -> str:
        return f"SensoComune({len(self.lemmas)} lemmi)"
    
    def __repr__(self) -> str:
        return f"SensoComune(lemmas={len(self.lemmas)})"
