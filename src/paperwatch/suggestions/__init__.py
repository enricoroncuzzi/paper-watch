"""Suggestion ranking and the checked 30-second summaries (design section 15)."""

from paperwatch.suggestions.config import SuggestionsConfig, load_suggestions
from paperwatch.suggestions.ranking import Candidate, Ranking, Scored, rank

__all__ = ["Candidate", "Ranking", "Scored", "SuggestionsConfig", "load_suggestions", "rank"]
