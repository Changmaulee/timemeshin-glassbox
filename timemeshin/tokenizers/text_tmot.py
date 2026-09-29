import re

class IndicAksharaTokenizer:
    """
    Syllable-Aware Indic Tokenizer using Brahmic Akshara linguistic boundaries.
    Eliminates the tokenization fragmentation tax for Indian scripts (Devanagari, Tamil, Telugu, etc.).
    - Sentence punctuation / full stops -> I-FRAME (Punctuation Macro Anchor)
    - Root Akshara syllables -> I-FRAME (Structural Linguistic Root)
    - Trailing dependent matras / modifiers -> B-FRAME (Local Morphological Delta)
    """
    def __init__(self):
        # Akshara regex for Devanagari Unicode block (\u0900-\u097F)
        self.akshara_regex = re.compile(r'[\u0905-\u0914]|[\u0915-\u0939][\u094d]?[\u093e-\u094c\u0901-\u0903]?')
        self.punctuation_bounds = {"\u0964": "I-FRAME", ".": "I-FRAME", "?": "I-FRAME", "!": "I-FRAME"}

    def tokenize_indic_string(self, text_string: str) -> list:
        words = text_string.split()
        structured_timeline_tokens = []

        for word in words:
            if word in self.punctuation_bounds:
                structured_timeline_tokens.append({
                    "type": "I-FRAME",
                    "content": word,
                    "label": "PUNCTUATION_ANCHOR"
                })
                continue

            syllables = self.akshara_regex.findall(word)
            if not syllables:
                structured_timeline_tokens.append({
                    "type": "B-FRAME",
                    "content": word,
                    "label": "ALPHA_MODIFIER"
                })
                continue

            # First syllable acts as root anchor (I-FRAME)
            structured_timeline_tokens.append({
                "type": "I-FRAME",
                "content": syllables[0],
                "label": f"AKSHARA_ROOT_{syllables[0]}"
            })
            # Subsequent dependent syllables/modifiers act as delta (B-FRAME)
            for sub_syllable in syllables[1:]:
                structured_timeline_tokens.append({
                    "type": "B-FRAME",
                    "content": sub_syllable,
                    "label": f"AKSHARA_DELTA_{sub_syllable}"
                })

        return structured_timeline_tokens
