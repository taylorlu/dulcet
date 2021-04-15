import librosa
from matplotlib.pyplot import axis
import numpy as np
from numpy.lib.twodim_base import tri
import tgt
import os
from typing import List, Union
from pathlib import Path


def vad():
    wav_path = r'D:\dataset\VCTK-Corpus\wav48\p227\p227_005.wav'
    y, sr = librosa.load(wav_path, sr=22050)

    slices = librosa.effects.split(y, top_db=20)

    trimed = []
    for slice in slices:
        trim = y[slice[0]:slice[1]]
        trimed.append(trim)

    trimed = np.concatenate(trimed, axis=-1)
    librosa.output.write_wav('local.wav', trimed, sr=22050)


PINYIN_DICT = {
    "a": ("^", "a"),
    "ai": ("^", "ai"),
    "an": ("^", "an"),
    "ang": ("^", "ang"),
    "ao": ("^", "ao"),
    "ba": ("b", "a"),
    "bai": ("b", "ai"),
    "ban": ("b", "an"),
    "bang": ("b", "ang"),
    "bao": ("b", "ao"),
    "be": ("b", "e"),
    "bei": ("b", "ei"),
    "ben": ("b", "en"),
    "beng": ("b", "eng"),
    "bi": ("b", "i"),
    "bian": ("b", "ian"),
    "biao": ("b", "iao"),
    "bie": ("b", "ie"),
    "bin": ("b", "in"),
    "bing": ("b", "ing"),
    "bo": ("b", "o"),
    "bu": ("b", "u"),
    "ca": ("c", "a"),
    "cai": ("c", "ai"),
    "can": ("c", "an"),
    "cang": ("c", "ang"),
    "cao": ("c", "ao"),
    "ce": ("c", "e"),
    "cen": ("c", "en"),
    "ceng": ("c", "eng"),
    "cha": ("ch", "a"),
    "chai": ("ch", "ai"),
    "chan": ("ch", "an"),
    "chang": ("ch", "ang"),
    "chao": ("ch", "ao"),
    "che": ("ch", "e"),
    "chen": ("ch", "en"),
    "cheng": ("ch", "eng"),
    "chi": ("ch", "iii"),
    "chong": ("ch", "ong"),
    "chou": ("ch", "ou"),
    "chu": ("ch", "u"),
    "chua": ("ch", "ua"),
    "chuai": ("ch", "uai"),
    "chuan": ("ch", "uan"),
    "chuang": ("ch", "uang"),
    "chui": ("ch", "uei"),
    "chun": ("ch", "uen"),
    "chuo": ("ch", "uo"),
    "ci": ("c", "ii"),
    "cong": ("c", "ong"),
    "cou": ("c", "ou"),
    "cu": ("c", "u"),
    "cuan": ("c", "uan"),
    "cui": ("c", "uei"),
    "cun": ("c", "uen"),
    "cuo": ("c", "uo"),
    "da": ("d", "a"),
    "dai": ("d", "ai"),
    "dan": ("d", "an"),
    "dang": ("d", "ang"),
    "dao": ("d", "ao"),
    "de": ("d", "e"),
    "dei": ("d", "ei"),
    "den": ("d", "en"),
    "deng": ("d", "eng"),
    "di": ("d", "i"),
    "dia": ("d", "ia"),
    "dian": ("d", "ian"),
    "diao": ("d", "iao"),
    "die": ("d", "ie"),
    "ding": ("d", "ing"),
    "diu": ("d", "iou"),
    "dong": ("d", "ong"),
    "dou": ("d", "ou"),
    "du": ("d", "u"),
    "duan": ("d", "uan"),
    "dui": ("d", "uei"),
    "dun": ("d", "uen"),
    "duo": ("d", "uo"),
    "e": ("^", "e"),
    "ei": ("^", "ei"),
    "en": ("^", "en"),
    "ng": ("^", "en"),
    "eng": ("^", "eng"),
    "er": ("^", "er"),
    "fa": ("f", "a"),
    "fan": ("f", "an"),
    "fang": ("f", "ang"),
    "fei": ("f", "ei"),
    "fen": ("f", "en"),
    "feng": ("f", "eng"),
    "fo": ("f", "o"),
    "fou": ("f", "ou"),
    "fu": ("f", "u"),
    "ga": ("g", "a"),
    "gai": ("g", "ai"),
    "gan": ("g", "an"),
    "gang": ("g", "ang"),
    "gao": ("g", "ao"),
    "ge": ("g", "e"),
    "gei": ("g", "ei"),
    "gen": ("g", "en"),
    "geng": ("g", "eng"),
    "gong": ("g", "ong"),
    "gou": ("g", "ou"),
    "gu": ("g", "u"),
    "gua": ("g", "ua"),
    "guai": ("g", "uai"),
    "guan": ("g", "uan"),
    "guang": ("g", "uang"),
    "gui": ("g", "uei"),
    "gun": ("g", "uen"),
    "guo": ("g", "uo"),
    "ha": ("h", "a"),
    "hai": ("h", "ai"),
    "han": ("h", "an"),
    "hang": ("h", "ang"),
    "hao": ("h", "ao"),
    "he": ("h", "e"),
    "hei": ("h", "ei"),
    "hen": ("h", "en"),
    "heng": ("h", "eng"),
    "hong": ("h", "ong"),
    "hou": ("h", "ou"),
    "hu": ("h", "u"),
    "hua": ("h", "ua"),
    "huai": ("h", "uai"),
    "huan": ("h", "uan"),
    "huang": ("h", "uang"),
    "hui": ("h", "uei"),
    "hun": ("h", "uen"),
    "huo": ("h", "uo"),
    "ji": ("j", "i"),
    "jia": ("j", "ia"),
    "jian": ("j", "ian"),
    "jiang": ("j", "iang"),
    "jiao": ("j", "iao"),
    "jie": ("j", "ie"),
    "jin": ("j", "in"),
    "jing": ("j", "ing"),
    "jiong": ("j", "iong"),
    "jiu": ("j", "iou"),
    "ju": ("j", "v"),
    "juan": ("j", "van"),
    "jue": ("j", "ve"),
    "jun": ("j", "vn"),
    "ka": ("k", "a"),
    "kai": ("k", "ai"),
    "kan": ("k", "an"),
    "kang": ("k", "ang"),
    "kao": ("k", "ao"),
    "ke": ("k", "e"),
    "kei": ("k", "ei"),
    "ken": ("k", "en"),
    "keng": ("k", "eng"),
    "kong": ("k", "ong"),
    "kou": ("k", "ou"),
    "ku": ("k", "u"),
    "kua": ("k", "ua"),
    "kuai": ("k", "uai"),
    "kuan": ("k", "uan"),
    "kuang": ("k", "uang"),
    "kui": ("k", "uei"),
    "kun": ("k", "uen"),
    "kuo": ("k", "uo"),
    "la": ("l", "a"),
    "lai": ("l", "ai"),
    "lan": ("l", "an"),
    "lang": ("l", "ang"),
    "lao": ("l", "ao"),
    "le": ("l", "e"),
    "lei": ("l", "ei"),
    "leng": ("l", "eng"),
    "li": ("l", "i"),
    "lia": ("l", "ia"),
    "lian": ("l", "ian"),
    "liang": ("l", "iang"),
    "liao": ("l", "iao"),
    "lie": ("l", "ie"),
    "lin": ("l", "in"),
    "ling": ("l", "ing"),
    "liu": ("l", "iou"),
    "lo": ("l", "o"),
    "long": ("l", "ong"),
    "lou": ("l", "ou"),
    "lu": ("l", "u"),
    "lv": ("l", "v"),
    "luan": ("l", "uan"),
    "lve": ("l", "ve"),
    "lue": ("l", "ve"),
    "lun": ("l", "uen"),
    "luo": ("l", "uo"),
    "ma": ("m", "a"),
    "mai": ("m", "ai"),
    "man": ("m", "an"),
    "mang": ("m", "ang"),
    "mao": ("m", "ao"),
    "me": ("m", "e"),
    "mei": ("m", "ei"),
    "men": ("m", "en"),
    "meng": ("m", "eng"),
    "mi": ("m", "i"),
    "mian": ("m", "ian"),
    "miao": ("m", "iao"),
    "mie": ("m", "ie"),
    "min": ("m", "in"),
    "ming": ("m", "ing"),
    "miu": ("m", "iou"),
    "mo": ("m", "o"),
    "mou": ("m", "ou"),
    "mu": ("m", "u"),
    "na": ("n", "a"),
    "nai": ("n", "ai"),
    "nan": ("n", "an"),
    "nang": ("n", "ang"),
    "nao": ("n", "ao"),
    "ne": ("n", "e"),
    "nei": ("n", "ei"),
    "nen": ("n", "en"),
    "neng": ("n", "eng"),
    "ni": ("n", "i"),
    "nia": ("n", "ia"),
    "nian": ("n", "ian"),
    "niang": ("n", "iang"),
    "niao": ("n", "iao"),
    "nie": ("n", "ie"),
    "nin": ("n", "in"),
    "ning": ("n", "ing"),
    "niu": ("n", "iou"),
    "nong": ("n", "ong"),
    "nou": ("n", "ou"),
    "nu": ("n", "u"),
    "nv": ("n", "v"),
    "nuan": ("n", "uan"),
    "nve": ("n", "ve"),
    "nue": ("n", "ve"),
    "nuo": ("n", "uo"),
    "o": ("^", "o"),
    "ou": ("^", "ou"),
    "pa": ("p", "a"),
    "pai": ("p", "ai"),
    "pan": ("p", "an"),
    "pang": ("p", "ang"),
    "pao": ("p", "ao"),
    "pe": ("p", "e"),
    "pei": ("p", "ei"),
    "pen": ("p", "en"),
    "peng": ("p", "eng"),
    "pi": ("p", "i"),
    "pian": ("p", "ian"),
    "piao": ("p", "iao"),
    "pie": ("p", "ie"),
    "pin": ("p", "in"),
    "ping": ("p", "ing"),
    "po": ("p", "o"),
    "pou": ("p", "ou"),
    "pu": ("p", "u"),
    "qi": ("q", "i"),
    "qia": ("q", "ia"),
    "qian": ("q", "ian"),
    "qiang": ("q", "iang"),
    "qiao": ("q", "iao"),
    "qie": ("q", "ie"),
    "qin": ("q", "in"),
    "qing": ("q", "ing"),
    "qiong": ("q", "iong"),
    "qiu": ("q", "iou"),
    "qu": ("q", "v"),
    "quan": ("q", "van"),
    "que": ("q", "ve"),
    "qun": ("q", "vn"),
    "ran": ("r", "an"),
    "rang": ("r", "ang"),
    "rao": ("r", "ao"),
    "re": ("r", "e"),
    "ren": ("r", "en"),
    "reng": ("r", "eng"),
    "ri": ("r", "iii"),
    "rong": ("r", "ong"),
    "rou": ("r", "ou"),
    "ru": ("r", "u"),
    "rua": ("r", "ua"),
    "ruan": ("r", "uan"),
    "rui": ("r", "uei"),
    "run": ("r", "uen"),
    "ruo": ("r", "uo"),
    "sa": ("s", "a"),
    "sai": ("s", "ai"),
    "san": ("s", "an"),
    "sang": ("s", "ang"),
    "sao": ("s", "ao"),
    "se": ("s", "e"),
    "sen": ("s", "en"),
    "seng": ("s", "eng"),
    "sha": ("sh", "a"),
    "shai": ("sh", "ai"),
    "shan": ("sh", "an"),
    "shang": ("sh", "ang"),
    "shao": ("sh", "ao"),
    "she": ("sh", "e"),
    "shei": ("sh", "ei"),
    "shen": ("sh", "en"),
    "sheng": ("sh", "eng"),
    "shi": ("sh", "iii"),
    "shou": ("sh", "ou"),
    "shu": ("sh", "u"),
    "shua": ("sh", "ua"),
    "shuai": ("sh", "uai"),
    "shuan": ("sh", "uan"),
    "shuang": ("sh", "uang"),
    "shui": ("sh", "uei"),
    "shun": ("sh", "uen"),
    "shuo": ("sh", "uo"),
    "si": ("s", "ii"),
    "song": ("s", "ong"),
    "sou": ("s", "ou"),
    "su": ("s", "u"),
    "suan": ("s", "uan"),
    "sui": ("s", "uei"),
    "sun": ("s", "uen"),
    "suo": ("s", "uo"),
    "ta": ("t", "a"),
    "tai": ("t", "ai"),
    "tan": ("t", "an"),
    "tang": ("t", "ang"),
    "tao": ("t", "ao"),
    "te": ("t", "e"),
    "tei": ("t", "ei"),
    "teng": ("t", "eng"),
    "ti": ("t", "i"),
    "tian": ("t", "ian"),
    "tiao": ("t", "iao"),
    "tie": ("t", "ie"),
    "ting": ("t", "ing"),
    "tong": ("t", "ong"),
    "tou": ("t", "ou"),
    "tu": ("t", "u"),
    "tuan": ("t", "uan"),
    "tui": ("t", "uei"),
    "tun": ("t", "uen"),
    "tuo": ("t", "uo"),
    "wa": ("^", "ua"),
    "wai": ("^", "uai"),
    "wan": ("^", "uan"),
    "wang": ("^", "uang"),
    "wei": ("^", "uei"),
    "wen": ("^", "uen"),
    "weng": ("^", "ueng"),
    "wo": ("^", "uo"),
    "wu": ("^", "u"),
    "xi": ("x", "i"),
    "xia": ("x", "ia"),
    "xian": ("x", "ian"),
    "xiang": ("x", "iang"),
    "xiao": ("x", "iao"),
    "xie": ("x", "ie"),
    "xin": ("x", "in"),
    "xing": ("x", "ing"),
    "xiong": ("x", "iong"),
    "xiu": ("x", "iou"),
    "xu": ("x", "v"),
    "xuan": ("x", "van"),
    "xue": ("x", "ve"),
    "xun": ("x", "vn"),
    "ya": ("^", "ia"),
    "yan": ("^", "ian"),
    "yang": ("^", "iang"),
    "yao": ("^", "iao"),
    "ye": ("^", "ie"),
    "yi": ("^", "i"),
    "yin": ("^", "in"),
    "ying": ("^", "ing"),
    "yo": ("^", "iou"),
    "yong": ("^", "iong"),
    "you": ("^", "iou"),
    "yu": ("^", "v"),
    "yuan": ("^", "van"),
    "yue": ("^", "ve"),
    "yun": ("^", "vn"),
    "za": ("z", "a"),
    "zai": ("z", "ai"),
    "zan": ("z", "an"),
    "zang": ("z", "ang"),
    "zao": ("z", "ao"),
    "ze": ("z", "e"),
    "zei": ("z", "ei"),
    "zen": ("z", "en"),
    "zeng": ("z", "eng"),
    "zha": ("zh", "a"),
    "zhai": ("zh", "ai"),
    "zhan": ("zh", "an"),
    "zhang": ("zh", "ang"),
    "zhao": ("zh", "ao"),
    "zhe": ("zh", "e"),
    "zhei": ("zh", "ei"),
    "zhen": ("zh", "en"),
    "zheng": ("zh", "eng"),
    "zhi": ("zh", "iii"),
    "zhong": ("zh", "ong"),
    "zhou": ("zh", "ou"),
    "zhu": ("zh", "u"),
    "zhua": ("zh", "ua"),
    "zhuai": ("zh", "uai"),
    "zhuan": ("zh", "uan"),
    "zhuang": ("zh", "uang"),
    "zhui": ("zh", "uei"),
    "zhun": ("zh", "uen"),
    "zhuo": ("zh", "uo"),
    "zi": ("z", "ii"),
    "zong": ("z", "ong"),
    "zou": ("z", "ou"),
    "zu": ("z", "u"),
    "zuan": ("z", "uan"),
    "zui": ("z", "uei"),
    "zun": ("z", "uen"),
    "zuo": ("z", "uo"),
}


vowels = set()
for tuples in PINYIN_DICT.values():
    for value in tuples:
        vowels.add(value)


# folder = r'D:\dataset\BZNSYP\PhoneLabeling'
# for file in os.listdir(folder):
#     grid = tgt.io.read_textgrid(os.path.join(folder, file))
#     intervals = grid.tiers[0].intervals
#     for interval in intervals:
#         if(''.join(list(interval.text)[-1]).isdigit()):
#             if(''.join(list(interval.text)[:-1]) not in vowels and ''.join(list(interval.text)[:-1])!='sp' and list(interval.text)[-2]!='r'):
#                 print(file)
#                 print(interval.text)
#         else:
#             if(interval.text not in vowels):
#                 if(interval.text=='sil'):
#                     continue
#                 if(interval.text.endswith('r')):
#                     continue
#                 print(interval.text)

def test_vowel():
    _alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')

    f = open('test.log', 'w')
    root = r'D:\dataset\VCTK-Corpus\txt'
    for folder in os.listdir(root):
        folder = os.path.join(root, folder)
        for txt in os.listdir(folder):
            txt = os.path.join(folder, txt)
            lines = open(txt).readlines()
            line = ''.join(lines)
            tokens = line.split()
            for token in tokens:
                token = list(token)
                if(token[0] not in _alphabet or token[-1] not in _alphabet):
                    # f.writelines(''.join(token)+"\n")
                    f.writelines(line+"\n")
                    break

# def get_files(path: Union[Path, str], extension='.wav') -> List[Path]:
#     """ Get all files from all subdirs with given extension. """
#     path = Path(path).expanduser().resolve()
#     return list(path.rglob(f'*{extension}'))

# all_wavs = get_files('wavs', extension='.wav')
# wav_paths = {w.with_suffix('').name: w for w in all_wavs}
# print(wav_paths)

# txt = r'D:\dataset\data_aishell3\train\content.txt'
# txt = open(txt, encoding='utf-8')
# lines = txt.readlines()
# for line in lines:
#     tokens = line.split()
#     wavname = tokens[0]
#     for token in tokens[2::2]:
#         print(token)
#     break

# from pathlib import Path
# from typing import Dict, List, Tuple, Union


# def get_files(path: Union[Path, str], extension='.wav') -> List[Path]:
#     """ Get all files from all subdirs with given extension. """
#     path = Path(path).expanduser().resolve()
#     return list(path.rglob(f'*{extension}'))

# all_wavs = get_files(Path(r'D:\dataset\LibriTTS\train-clean-100\19'), extension='.wav')
# for wav in all_wavs:
#     txt = wav.with_suffix('.normalized.txt')
#     print(txt)

from p_tqdm import p_uimap
from typing import Tuple

def process_wav(wav_tuple: Tuple):
    fullpath, data_type = wav_tuple
    return fullpath

dic = {}
dic['name1'] = [1,2]
dic['name2'] = [8,2]
dic['name3'] = [9,2]

wav_iter = p_uimap(process_wav, dic.values())
for out_dict in wav_iter:
    print(out_dict)
