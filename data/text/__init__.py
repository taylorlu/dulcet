from typing import Union

from data.text.symbols import all_phonemes
from data.text.tokenizer import Phonemizer, Tokenizer


class TextToTokens:
    def __init__(self, phonemizer: Phonemizer, tokenizer: Tokenizer):
        self.phonemizer = phonemizer
        self.tokenizer = tokenizer
    
    def __call__(self, input_text: Union[str, list], language=None) -> list:
        phons = self.phonemizer(input_text, language=language)
        if(phons.startswith('{')):
            phons = phons[1:]
        if(phons.endswith('}')):
            phons = phons[:-1]
        tokens = self.tokenizer(phons)
        return tokens
    
    @classmethod
    def default(cls, english_lexicon_path, pinyin_lexicon_path, add_start_end: bool):
        phonemizer = Phonemizer(english_lexicon_path, pinyin_lexicon_path)
        tokenizer = Tokenizer(add_start_end=add_start_end)
        return cls(phonemizer=phonemizer, tokenizer=tokenizer)
