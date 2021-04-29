from typing import Union
import re
import sys
import os
sys.path.append(os.getcwd())
from string import punctuation
from g2p_en import G2p
from pypinyin import pinyin, Style
from data.text.symbols import all_phonemes


class Tokenizer:
    
    def __init__(self, start_token='>', end_token='<', pad_token='/', add_start_end=False):
        self.alphabet = all_phonemes
        self.idx_to_token = {i: s for i, s in enumerate(self.alphabet, start=1)}
        self.idx_to_token[0] = pad_token
        self.token_to_idx = {s: [i] for i, s in self.idx_to_token.items()}
        self.vocab_size = len(self.alphabet) + 1
        self.add_start_end = add_start_end
        if add_start_end:
            self.start_token_index = len(self.alphabet) + 1
            self.end_token_index = len(self.alphabet) + 2
            self.vocab_size += 2
            self.idx_to_token[self.start_token_index] = start_token
            self.idx_to_token[self.end_token_index] = end_token
    
    def __call__(self, sentence: str) -> list:
        sequence = [self.token_to_idx[c] for c in sentence.split()]  # No filtering: text should only contain known chars.
        sequence = [item for items in sequence for item in items]
        if self.add_start_end:
            sequence = [self.start_token_index] + sequence + [self.end_token_index]
        return sequence
    
    def decode(self, sequence: list) -> str:
        return ''.join([self.idx_to_token[int(t)] for t in sequence])


class Phonemizer:

    def __init__(self):
        english_lexicon_path = r'data\text\lexicon\librispeech-lexicon.txt'
        pinyin_lexicon_path = r'data\text\lexicon\pinyin-lexicon-r.txt'
        self.english_lexicon = self.read_lexicon(english_lexicon_path)
        self.pinyin_lexicon = self.read_lexicon(pinyin_lexicon_path)
        self.g2p = G2p()

    def read_lexicon(self, lex_path):
        lexicon = {}
        with open(lex_path) as f:
            for line in f:
                temp = re.split(r"\s+", line.strip("\n"))
                word = temp[0]
                phones = temp[1:]
                if word.lower() not in lexicon:
                    lexicon[word.lower()] = phones
        return lexicon

    def preprocess_english(self, text):
        text = text.rstrip(punctuation)

        phones = []
        words = re.split(r"([,;.\-\?\!\s+])", text)
        for w in words:
            if w.lower() in self.english_lexicon:
                phones += self.english_lexicon[w.lower()]
            else:
                phones += list(filter(lambda p: p != " ", self.g2p(w)))
        phones = "{" + "}{".join(phones) + "}"
        phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
        phones = phones.replace("}{", " ")

        return phones

    def preprocess_mandarin(self, text):
        phones = []
        pinyins = [
            p[0]
            for p in pinyin(
                text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
            )
        ]
        for p in pinyins:
            if p in self.pinyin_lexicon:
                phones += self.pinyin_lexicon[p]
            else:
                phones.append("sp")

        phones = "{" + " ".join(phones) + "}"
        return phones

    def preprocess_pinyin(self, text):
        phones = []
        pinyins = text.split()
        for p in pinyins:
            if p in self.pinyin_lexicon:
                phones += self.pinyin_lexicon[p]
            else:
                phones.append("sp")

        phones = "{" + " ".join(phones) + "}"
        return phones

    def __call__(self, text: Union[str, list], language=None) -> Union[str, list]:
        # phonemizer does not like hyphens.
        if(language=='english'):
            phonemes = self.preprocess_english(text)
        elif(language=='mandarin'):
            phonemes = self.preprocess_mandarin(text)
        elif(language=='pinyin'):
            phonemes = self.preprocess_pinyin(text)
        else:   # default mandarin
            phonemes = self.preprocess_mandarin(text)
        return phonemes


if(__name__=='__main__'):
    phonemer = Phonemizer()
    phones = phonemer.preprocess_english('Internally DSAlign uses the DeepSpeech STT engine. For it to be able to function, it requires')
    print(phones)
    phones = phonemer.preprocess_mandarin('新增较多镜头，详见平面稿及附件')
    print(phones)
    phones = phonemer.preprocess_pinyin('lv4 shi4 yang2, chun1')
    print(phones)
