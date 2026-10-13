import math
import re
from typing import Dict, List, Optional
import numpy as np
from core.models import CommunicationMessage, EmployeeContext, MarkerVector
from extractors.base import IFeatureExtractor


class SpacyStatisticalExtractor(IFeatureExtractor):
    """
    Statistical NLP extractor implementing linguistic indices from NIR-1 and NIR-2:
    - Pronoun focus: ratio of 1st person singular (I, me, my) to 1st person plural (we, us, our);
    - Syntactic complexity: mean parse tree depth and punctuation irregularity;
    - POS Shannon entropy: distribution uniformity of grammatical word classes.
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
    PUNCT_PATTERN = re.compile(r"[!?]{2,}|\.{3,}")

    def __init__(self, spacy_model: Optional[str] = None):
        self.nlp = None
        if spacy_model:
            try:
                import spacy
                self.nlp = spacy.load(spacy_model)
            except Exception:
                self.nlp = None

    def compute_pronoun_marker(self, text: str) -> float:
        """
        Calculate self-focus pronoun ratio: |I-words| / (|I-words| + |We-words| + 1).
        Ranges in [0.0, 1.0]. High values denote egocentric detachment.
        """
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
        Estimate syntactic complexity using spaCy dependency tree depth
        or fallback structural proxy based on clause length and punctuation density.
        """
        if not text.strip():
            return 0.0

        if self.nlp is not None:
            doc = self.nlp(text[:2000])
            depths: List[int] = []
            for sent in doc.sents:
                root = sent.root
                depth = self._get_subtree_depth(root)
                depths.append(depth)
            mean_depth = float(sum(depths) / max(len(depths), 1))
            norm_depth = float(np.clip((mean_depth - 2.0) / 8.0, 0.0, 1.0))
            return norm_depth
        clauses = re.split(r"[,;:\n.]+", text)
        clean_clauses = [c.strip() for c in clauses if len(c.strip()) > 0]
        if not clean_clauses:
            return 0.0
        clause_lengths = [len(c.split()) for c in clean_clauses]
        mean_len = sum(clause_lengths) / len(clause_lengths)
        norm_len = float(np.clip(mean_len / 25.0, 0.0, 1.0))
        return norm_len

    def _get_subtree_depth(self, node) -> int:
        """
        Recursively compute syntactic dependency tree depth.
        """
        if not list(node.children):
            return 1
        return 1 + max(self._get_subtree_depth(child) for child in node.children)

    def compute_pos_entropy(self, text: str) -> float:
        """
        Calculate Shannon entropy over POS tag distribution: H(POS) = -sum p_i * log2(p_i).
        Normalized into [0.0, 1.0] by log2(|Categories|).
        """
        if self.nlp is None or not text.strip():
            return 0.5
        doc = self.nlp(text[:2000])
        pos_counts: Dict[str, int] = {}
        total = 0
        for token in doc:
            if not token.is_space and not token.is_punct:
                pos = token.pos_
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
                total += 1
        if total < 5:
            return 0.5
        entropy = 0.0
        for count in pos_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        max_entropy = math.log2(17)
        return float(np.clip(entropy / max_entropy, 0.0, 1.0))

    def compute_markers(self, message: CommunicationMessage, context: EmployeeContext) -> MarkerVector:
        """
        Aggregate statistical NLP markers into unified MarkerVector.
        """
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
            vector.response_marker = float(np.clip(rep_std / 24.0, 0.0, 1.0))
            vector.social_marker = float(np.clip(1.0 - context.reciprocity, 0.0, 1.0))
        return vector
