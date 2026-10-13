import math
import re
from typing import Optional
import numpy as np
from core.models import CommunicationMessage, EmployeeContext, MarkerVector
from extractors.base import IFeatureExtractor


class SpacyStatisticalExtractor(IFeatureExtractor):
    """
    Statistical NLP extractor implementing linguistic indices from NIR-1 and NIR-2:
    - Pronoun ratio: self-focus detachment;
    - Syntactic complexity: punctuation density and clause structure;
    - Informational variance and latency dynamics.
    """

    FIRST_PERSON_SINGULAR = {
        "i", "me", "my", "mine", "myself",
        "я", "меня", "мне", "мной", "мною", "мой", "моя", "мое", "мои"
    }

    FIRST_PERSON_PLURAL = {
        "we", "us", "our", "ours", "ourselves",
        "мы", "нас", "нам", "нами", "наш", "наша", "наше", "наши"
    }

    TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
    PUNCT_PATTERN = re.compile(r"[,;:\.\?!-]")

    def __init__(self, spacy_model: Optional[str] = None):
        self.nlp = None
        if spacy_model:
            try:
                import spacy
                self.nlp = spacy.load(spacy_model)
            except Exception:
                self.nlp = None

    def compute_pronoun_marker(self, text: str) -> float:
        tokens = [t.lower() for t in self.TOKEN_PATTERN.findall(text)]
        if not tokens:
            return 0.0
        i_count = sum(1 for t in tokens if t in self.FIRST_PERSON_SINGULAR)
        we_count = sum(1 for t in tokens if t in self.FIRST_PERSON_PLURAL)
        if i_count + we_count == 0:
            return 0.0
        return float(i_count / (i_count + we_count + 1.0))

    def compute_syntactic_complexity(self, text: str) -> float:
        """
        Syntactic complexity based on punctuation density and clause structure.
        Ranges stably between 0.1 and 0.8 without hard saturation.
        """
        words = self.TOKEN_PATTERN.findall(text)
        if not words:
            return 0.25
        num_words = len(words)
        num_punct = len(self.PUNCT_PATTERN.findall(text))
        punct_density = num_punct / max(num_words, 1)
        length_factor = math.log10(max(num_words, 2)) / 3.0
        raw_score = 0.5 * punct_density * 5.0 + 0.5 * length_factor
        return float(np.clip(raw_score, 0.10, 0.85))

    def compute_markers(self, message: CommunicationMessage, context: Optional[EmployeeContext] = None) -> MarkerVector:
        text = message.text_content
        vector = MarkerVector()
        if not text:
            return vector
        vector.pronoun_marker = self.compute_pronoun_marker(text)
        vector.syntactic_marker = self.compute_syntactic_complexity(text)
        if context is not None:
            act_change = abs(context.daily_activity_change)
            vector.activity_marker = float(np.clip(act_change / 3.0, 0.0, 1.0))
            rep_std = context.reply_time_std
            vector.response_marker = float(np.clip(rep_std / 48.0, 0.0, 1.0))
            recip = context.reciprocity
            vector.social_marker = float(np.clip(1.0 - recip, 0.0, 1.0))
        return vector