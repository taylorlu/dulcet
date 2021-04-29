""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """
import os, sys
sys.path.append(os.getcwd())
from data.text import cmudict, pinyin

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["sp", "spn", "sil"]

# Export all symbols:
all_phonemes = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + cmudict.valid_symbols
    + pinyin.valid_symbols
    + _silences
)
