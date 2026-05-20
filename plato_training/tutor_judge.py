"""
TutorJudge — TUTOR language answer-judging primitives (1965→2026)

Bridges PLATO's TUTOR answer judging with modern semantic retrieval.
Three-tier cascade: exact → bitvector → semantic.

CPU only, no GPU required.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Part 1: Bit-Vector Fingerprint (TUTOR 1965)
# ---------------------------------------------------------------------------

def word_to_bitvector(word: str) -> int:
    """
    TUTOR-style word fingerprint. 64-bit integer with:
    - Bits 0-25:  letter presence (a=0, b=1, ..., z=25)
    - Bits 26-51: letter pair hashes
    - Bits 52-57: first letter position
    - Bits 58-63: length indicator
    """
    bits = 0
    word = word.lower().strip()
    if not word:
        return 0
    # Letter presence
    for ch in word:
        if 'a' <= ch <= 'z':
            bits |= (1 << (ord(ch) - ord('a')))
    # Letter pair hashes
    pair_bit = 26
    for i in range(min(len(word) - 1, 26)):
        pair = word[i:i + 2]
        if pair_bit < 52:
            # Deterministic hash from character pair
            h = (ord(pair[0]) * 31 + ord(pair[1])) & 1
            bits |= (h << pair_bit)
            pair_bit += 1
    # First letter position
    if 'a' <= word[0] <= 'z':
        bits |= ((ord(word[0]) - ord('a')) << 52)
    # Length indicator
    bits |= (min(len(word), 63) << 58)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Popcount of XOR — number of differing bits."""
    return bin(a ^ b).count('1')


def word_similarity(word_a: str, word_b: str) -> float:
    """Similarity in [0, 1] based on bit-vector Hamming distance."""
    if not word_a or not word_b:
        return 0.0
    bv_a = word_to_bitvector(word_a)
    bv_b = word_to_bitvector(word_b)
    if bv_a == 0 and bv_b == 0:
        return 0.0
    return 1.0 - (hamming_distance(bv_a, bv_b) / 64.0)


# ---------------------------------------------------------------------------
# Part 2: Pattern Parsing + Concept + Numerical Judging
# ---------------------------------------------------------------------------

@dataclass
class JudgeSpecs:
    """Adjustable matching strictness (TUTOR specs command)."""
    spelling_tolerance: int = 2        # max char differences
    ignore_words: set = field(default_factory=lambda: {"a", "an", "the", "it", "is", "its"})
    noorder: bool = False
    min_bitvector_similarity: float = 0.88
    min_semantic_similarity: float = 0.60


def _parse_pattern(pattern_str: str):
    """
    Parse TUTOR answer pattern syntax into a structured form.

    Syntax:
      <word, word, ...>   → optional words
      (word, word, ...)   → alternative groups (any one matches)
      bare_word           → required word

    Returns list of: ('required', word) | ('optional', [words]) | ('alt', [words])
    """
    tokens = []
    i = 0
    s = pattern_str.strip()
    while i < len(s):
        if s[i] == '<':
            # Optional group
            end = s.index('>', i)
            words = [w.strip() for w in s[i + 1:end].split(',') if w.strip()]
            tokens.append(('optional', words))
            i = end + 1
        elif s[i] == '(':
            end = s.index(')', i)
            words = [w.strip() for w in s[i + 1:end].split(',') if w.strip()]
            tokens.append(('alt', words))
            i = end + 1
        elif s[i] in (' ', ','):
            i += 1
        else:
            # Required word
            j = i
            while j < len(s) and s[j] not in (' ', '<', '(', ','):
                j += 1
            word = s[i:j].strip()
            if word:
                tokens.append(('required', word))
            i = j
    return tokens


# Common English stopwords that inflate bitvector similarity
_BITVECTOR_STOPWORDS = frozenset({
    "what", "is", "the", "a", "an", "how", "does", "do", "tell", "me",
    "about", "it", "that", "this", "of", "for", "in", "on", "to", "and", "or",
})


def _tokenize_response(response: str, ignore_words: set | None = None) -> list[str]:
    """Split response into words, stripping ignored words."""
    if ignore_words is None:
        ignore_words = set()
    words = re.findall(r"[a-zA-Z0-9./\-]+", response.lower())
    return [w for w in words if w not in {iw.lower() for iw in ignore_words}]


def _levenshtein(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(
                prev[j + 1] + 1,   # delete
                curr[j] + 1,       # insert
                prev[j] + (0 if ca == cb else 1),  # substitute
            ))
        prev = curr
    return prev[-1]


def _parse_number(s: str) -> Optional[float]:
    """Parse a number string, including fractions and 'pi'."""
    s = s.strip().lower()
    if not s:
        return None

    # Pi variants
    if s in ('pi', 'π'):
        return math.pi

    # Fraction: "22/7"
    if '/' in s:
        parts = s.split('/')
        if len(parts) == 2:
            try:
                num, den = float(parts[0]), float(parts[1])
                if den != 0:
                    return num / den
            except ValueError:
                return None

    # Number with pi: "3pi", "2*pi"
    pi_match = re.match(r'^([0-9.]+)\s*\*?\s*pi$', s)
    if pi_match:
        try:
            return float(pi_match.group(1)) * math.pi
        except ValueError:
            return None

    # Plain number
    try:
        return float(s)
    except ValueError:
        return None


class TutorJudge:
    """
    TUTOR-style answer judge with three-tier cascade:
    exact → bitvector → semantic.
    """

    def __init__(self, specs: JudgeSpecs | None = None):
        self.specs = specs or JudgeSpecs()
        self.concepts: dict[str, set[str]] = {}
        self._semantic_model = None
        self._semantic_available: bool | None = None

    # -- Concept management --

    def add_concept(self, name: str, words: list[str] | set[str]) -> None:
        """Add a vocabulary cluster (TUTOR concept/vocabs)."""
        self.concepts[name] = {w.lower().strip() for w in words if w.strip()}

    # -- Semantic model (lazy load) --

    def _ensure_semantic(self):
        """Try to load Model2Vec for semantic fallback."""
        if self._semantic_available is not None:
            return
        try:
            from model2vec import StaticModel
            self._semantic_model = StaticModel.from_pretrained("minishlab/potion-base-8M")
            self._semantic_available = True
        except Exception:
            self._semantic_model = None
            self._semantic_available = False

    def _semantic_similarity(self, a: str, b: str) -> float:
        """Compute semantic similarity between two strings."""
        self._ensure_semantic()
        if not self._semantic_available or not self._semantic_model:
            return 0.0
        try:
            vecs = self._semantic_model.encode([a, b])
            dot = sum(v1 * v2 for v1, v2 in zip(vecs[0], vecs[1]))
            norm_a = math.sqrt(sum(v * v for v in vecs[0]))
            norm_b = math.sqrt(sum(v * v for v in vecs[1]))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)
        except Exception:
            return 0.0

    # -- Main judge --

    def judge(
        self,
        response: str,
        correct_patterns: list[str] | None = None,
        wrong_patterns: list[str] | None = None,
    ) -> dict:
        """
        Judge a student response against correct/wrong patterns.

        Returns dict with:
          correct: bool
          similarity: float (0-1)
          method: str ('exact', 'bitvector', 'semantic', 'concept', 'none')
          matched_pattern: str | None
        """
        result = {
            'correct': False,
            'similarity': 0.0,
            'method': 'none',
            'matched_pattern': None,
        }

        if correct_patterns is None:
            correct_patterns = []
        if wrong_patterns is None:
            wrong_patterns = []

        resp_words = _tokenize_response(response, self.specs.ignore_words)

        # Check wrong patterns first
        for wp in wrong_patterns:
            if self._match_pattern(resp_words, wp)['matched']:
                result['correct'] = False
                result['method'] = 'wrong_pattern'
                result['matched_pattern'] = wp
                return result

        # Check correct patterns — cascade through methods
        best = result
        for pattern in correct_patterns:
            match = self._judge_pattern_cascade(resp_words, pattern)
            if match['similarity'] > best['similarity']:
                best = match
                if best['correct']:
                    break

        # Check concepts if no pattern match
        if not best['correct']:
            concept_match = self._check_concepts(resp_words)
            if concept_match['similarity'] > best['similarity']:
                best = concept_match

        return best

    def _judge_pattern_cascade(self, resp_words: list[str], pattern: str) -> dict:
        """Three-tier cascade: exact → bitvector → semantic."""
        # Tier 1: Exact
        exact = self._match_pattern(resp_words, pattern)
        if exact['matched']:
            return {
                'correct': True,
                'similarity': 1.0,
                'method': 'exact',
                'matched_pattern': pattern,
            }

        # Tier 2: Bitvector (fuzzy matching on words)
        bv_sim = self._bitvector_pattern_similarity(resp_words, pattern)
        if bv_sim >= self.specs.min_bitvector_similarity:
            return {
                'correct': True,
                'similarity': bv_sim,
                'method': 'bitvector',
                'matched_pattern': pattern,
            }

        # Tier 3: Semantic
        sem_sim = self._semantic_similarity(
            ' '.join(resp_words),
            self._pattern_to_plain_text(pattern),
        )
        if sem_sim >= self.specs.min_semantic_similarity:
            return {
                'correct': True,
                'similarity': sem_sim,
                'method': 'semantic',
                'matched_pattern': pattern,
            }

        return {
            'correct': False,
            'similarity': max(bv_sim, sem_sim),
            'method': 'none',
            'matched_pattern': pattern,
        }

    def _match_pattern(self, resp_words: list[str], pattern: str) -> dict:
        """Exact match against TUTOR pattern syntax."""
        tokens = _parse_pattern(pattern)
        return self._match_tokens(resp_words, tokens)

    def _match_tokens(self, resp_words: list[str], tokens: list) -> dict:
        """Match response words against parsed pattern tokens."""
        if self.specs.noorder:
            return self._match_tokens_noorder(resp_words, tokens)

        # Ordered matching with greedy optional/alt
        return self._match_tokens_ordered(resp_words, tokens)

    def _match_tokens_ordered(self, resp_words: list[str], tokens: list) -> dict:
        """Ordered pattern matching. Optional/alt words are flexible."""
        ri = 0  # response index
        for tok in tokens:
            kind, val = tok
            if kind == 'optional':
                # Skip if current response word matches one of the optional words
                if ri < len(resp_words) and resp_words[ri] in [w.lower() for w in val]:
                    ri += 1
                # Optional — skip either way
            elif kind == 'alt':
                # Must match one of the alternatives
                if ri >= len(resp_words):
                    return {'matched': False}
                if resp_words[ri] in [w.lower() for w in val]:
                    ri += 1
                else:
                    return {'matched': False}
            elif kind == 'required':
                if ri >= len(resp_words):
                    return {'matched': False}
                if resp_words[ri] == val.lower():
                    ri += 1
                else:
                    # Check if it's in remaining words (extra words tolerance)
                    found = False
                    for j in range(ri, len(resp_words)):
                        if resp_words[j] == val.lower():
                            ri = j + 1
                            found = True
                            break
                    if not found:
                        return {'matched': False}
        return {'matched': ri >= 0}  # all tokens consumed or extra words ok

    def _match_tokens_noorder(self, resp_words: list[str], tokens: list) -> dict:
        """Order-independent matching. All required/alt present, optional ignored."""
        resp_set = set(resp_words)
        for tok in tokens:
            kind, val = tok
            if kind == 'required':
                if val.lower() not in resp_set:
                    return {'matched': False}
            elif kind == 'alt':
                if not any(w.lower() in resp_set for w in val):
                    return {'matched': False}
            # optional — always ok with noorder
        return {'matched': True}

    def _bitvector_pattern_similarity(self, resp_words: list[str], pattern: str) -> float:
        """Best bitvector similarity between response and pattern.

        Stopwords are filtered out so common function words ("what", "is",
        "the", ...) don't inflate similarity scores.
        """
        tokens = _parse_pattern(pattern)
        # Flatten pattern into required + alternative words
        pattern_words = []
        for kind, val in tokens:
            if kind == 'required':
                w = val.lower()
                if w not in _BITVECTOR_STOPWORDS:
                    pattern_words.append(w)
            elif kind == 'alt':
                for w in val:
                    w = w.lower()
                    if w not in _BITVECTOR_STOPWORDS:
                        pattern_words.append(w)
            elif kind == 'optional':
                for w in val:
                    w = w.lower()
                    if w not in _BITVECTOR_STOPWORDS:
                        pattern_words.append(w)

        # Filter stopwords from response words too
        resp_content = [w for w in resp_words if w not in _BITVECTOR_STOPWORDS]

        if not pattern_words or not resp_content:
            return 0.0

        # For each response word, find best match in pattern
        total_sim = 0.0
        for rw in resp_content:
            best = 0.0
            for pw in pattern_words:
                sim = word_similarity(rw, pw)
                # Also check Levenshtein for short words
                if len(rw) <= 6 and len(pw) <= 6:
                    dist = _levenshtein(rw, pw)
                    lev_sim = 1.0 - (dist / max(len(rw), len(pw)))
                    sim = max(sim, lev_sim)
                best = max(best, sim)
            total_sim += best

        return total_sim / max(len(resp_content), len(pattern_words))

    def _pattern_to_plain_text(self, pattern: str) -> str:
        """Convert pattern to plain text for semantic comparison."""
        return re.sub(r'[<>()]', ' ', pattern).replace(',', ' ').strip()
        # Collapse whitespace
        return ' '.join(_pattern_to_plain_text_result.split())

    def _check_concepts(self, resp_words: list[str]) -> dict:
        """Check if response words match any concept."""
        best_sim = 0.0
        best_concept = None
        for cname, cwords in self.concepts.items():
            for rw in resp_words:
                for cw in cwords:
                    sim = word_similarity(rw, cw)
                    if sim > best_sim:
                        best_sim = sim
                        best_concept = cname
        if best_sim >= self.specs.min_bitvector_similarity:
            return {
                'correct': True,
                'similarity': best_sim,
                'method': 'concept',
                'matched_pattern': f'concept:{best_concept}',
            }
        return {
            'correct': False,
            'similarity': best_sim,
            'method': 'none',
            'matched_pattern': None,
        }

    # -- Numerical judging --

    def judge_numerical(
        self,
        response: str,
        correct_value: float,
        tolerance: float = 0.01,
    ) -> dict:
        """
        Judge a numerical response (TUTOR ansv).
        Handles fractions, pi, plain numbers.
        """
        parsed = _parse_number(response)
        if parsed is None:
            # Try extracting any number from the response
            nums = re.findall(r'-?[0-9.]+', response)
            if nums:
                try:
                    parsed = float(nums[0])
                except ValueError:
                    return {
                        'correct': False,
                        'similarity': 0.0,
                        'method': 'numerical',
                        'parsed': None,
                    }
            else:
                return {
                    'correct': False,
                    'similarity': 0.0,
                    'method': 'numerical',
                    'parsed': None,
                }

        diff = abs(parsed - correct_value)
        max_diff = max(abs(parsed), abs(correct_value), 1.0)
        similarity = max(0.0, 1.0 - (diff / max_diff))

        return {
            'correct': diff <= tolerance,
            'similarity': similarity,
            'method': 'numerical',
            'parsed': parsed,
        }


# ---------------------------------------------------------------------------
# Part 3: Benchmark
# ---------------------------------------------------------------------------

def run_benchmark() -> dict:
    """Compare exact vs bitvector vs semantic on test pairs."""
    judge = TutorJudge()

    test_pairs = [
        # (response, correct_word, expected_match)
        ("triangle", "triangle", True),
        ("triangel", "triangle", True),       # typo
        ("triangl", "triangle", True),         # typo
        ("right triangle", "triangle", True),  # extra word
        ("square", "triangle", False),
        ("circle", "triangle", False),
        ("polygon", "triangle", False),
        ("equilateral", "triangle", False),    # related but different
        ("hypotenuse", "triangle", False),
        ("trinagle", "triangle", True),        # transposition typo
        ("abc", "def", False),
        ("", "triangle", False),
        ("6", "6", True),
        ("six", "6", False),                   # semantic gap
        ("pi", "3.14159", True),               # numerical
        ("3.14", "3.14159", True),
        ("22/7", "3.14159", True),
        ("rectangle", "rectangle", True),
        ("rectangel", "rectangle", True),
        ("squar", "square", True),
        ("circel", "circle", True),
        ("pentagn", "pentagon", True),
        ("hexagon", "hexagon", True),
        ("heptagn", "heptagon", True),
        ("octagon", "octagon", True),
        (" parallagram", "parallelogram", True),
        ("rhombus", "rhombus", True),
        ("rhomus", "rhombus", True),
        ("kite", "kite", True),
        ("trapezoid", "trapezoid", True),
        ("trapezium", "trapezium", True),
        ("isosceles", "isosceles", True),
        ("isoscoles", "isosceles", True),
        ("scalene", "scalene", True),
        ("sclene", "scalene", True),
        ("vertex", "vertex", True),
        ("vortex", "vertex", True),            # phonetic confusion
        ("angle", "angle", True),
        ("angel", "angle", True),              # real word but typo context
        ("perimeter", "perimeter", True),
        ("area", "area", True),
        ("volume", "volume", True),
        ("diameter", "diameter", True),
        ("radius", "radius", True),
        ("circumference", "circumference", True),
        ("circumfrance", "circumference", True),
        ("congruent", "congruent", True),
        ("similar", "similar", True),
    ]

    exact_hits = 0
    bv_hits = 0
    sem_hits = 0
    total = len(test_pairs)

    for response, correct, expected in test_pairs:
        resp_clean = response.strip().lower()

        # Exact
        if resp_clean == correct.lower():
            exact_hits += 1

        # Bitvector
        sim = word_similarity(resp_clean, correct.lower())
        if sim >= 0.80 and expected:
            bv_hits += 1
        elif sim < 0.80 and not expected:
            bv_hits += 1  # correct rejection

        # Semantic (if available)
        judge._ensure_semantic()
        if judge._semantic_available:
            sem_sim = judge._semantic_similarity(resp_clean, correct.lower())
            if (sem_sim >= 0.60 and expected) or (sem_sim < 0.60 and not expected):
                sem_hits += 1

    results = {
        'total': total,
        'exact_accuracy': exact_hits / total,
        'bitvector_accuracy': bv_hits / total,
        'bitvector_available': True,
    }

    if judge._semantic_available:
        results['semantic_accuracy'] = sem_hits / total
        results['semantic_available'] = True
    else:
        results['semantic_available'] = False

    return results
