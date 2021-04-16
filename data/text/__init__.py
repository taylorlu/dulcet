from typing import Union

from data.text.symbols import all_phonemes
from data.text.tokenizer import Phonemizer, Tokenizer


class TextToTokens:
    def __init__(self, phonemizer: Phonemizer, tokenizer: Tokenizer):
        self.phonemizer = phonemizer
        self.tokenizer = tokenizer
    
    def __call__(self, input_text: Union[str, list]) -> list:
        phons = self.phonemizer(input_text)
        tokens = self.tokenizer(phons)
        return tokens
    
    @classmethod
    def default(cls, add_start_end: bool, with_stress: bool, njobs=1):
        phonemizer = Phonemizer()
        tokenizer = Tokenizer(add_start_end=add_start_end)
        return cls(phonemizer=phonemizer, tokenizer=tokenizer)
