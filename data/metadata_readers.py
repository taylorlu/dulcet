"""
    methods for reading a dataset and return a dictionary of the form:
    {
      filename: text_line,
      ...
    }
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union


def get_files(path: Union[Path, str], extension='.wav') -> List[Path]:
    """ Get all files from all subdirs with given extension. """
    path = Path(path).expanduser().resolve()
    return list(path.rglob(f'*{extension}'))


def get_preprocessor_by_name(name: str):
    """
    Returns the respective data function.
    Taken from https://github.com/mozilla/TTS/blob/master/TTS/tts/datasets/preprocess.py
    """
    thismodule = sys.modules[__name__]
    return getattr(thismodule, name.lower())


def ljspeech(metadata_path: str, column_sep='|') -> dict:
    text_dict = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            l_split = l.split(column_sep)
            filename, text = l_split[0], l_split[-1]
            if filename.endswith('.wav'):
                filename = filename.split('.')[0]
            text = text.replace('\n', '')
            text_dict.update({filename: text})
    return text_dict


def vctk(dataset_dir: str) -> dict:
    text_dict = {}
    all_wavs = get_files(Path(dataset_dir) / 'wav48', extension='.wav')
    wav_paths = {w.stem: w for w in all_wavs}
    for name in wav_paths:
        txt_path = Path(dataset_dir) / 'txt' / name.split('_')[0] / (name+'.txt')
        if(txt_path.is_file()):
            text = open(txt_path).readline()
            text_dict.update({name: (wav_paths[name], 'vctk', name.split('_')[0], text.strip())})

    return text_dict


def libritts_100(dataset_dir: str) -> dict:
    text_dict = {}
    all_wavs = get_files(Path(dataset_dir) / 'train-clean-100', extension='.wav')
    wav_paths = {w.stem: w for w in all_wavs}
    for name in wav_paths:
        txt_path = wav_paths[name].with_suffix('.normalized.txt')
        if(txt_path.is_file()):
            text = open(txt_path, encoding='utf-8').readline()
            text_dict.update({name: (wav_paths[name], 'libritts_100', name.split('_')[0], text.strip())})

    return text_dict


def libritts_360(dataset_dir: str) -> dict:
    text_dict = {}
    all_wavs = get_files(Path(dataset_dir) / 'train-clean-360', extension='.wav')
    wav_paths = {w.stem: w for w in all_wavs}
    for name in wav_paths:
        txt_path = wav_paths[name].with_suffix('.normalized.txt')
        if(txt_path.is_file()):
            text = open(txt_path, encoding='utf-8').readline()
            text_dict.update({name: (wav_paths[name], 'libritts_360', name.split('_')[0], text.strip())})

    return text_dict


def aishell3(dataset_dir: str) -> dict:
    text_dict = {}
    subFolders = ['train', 'test']

    for subFolder in subFolders:
        label_path = Path(dataset_dir) / subFolder / 'content.txt'
        labelfile = open(str(label_path), encoding='utf-8')

        while(True):
            line=labelfile.readline().strip()
            if(line==None or len(line)<1):
                break

            tokens = line.split()
            wavname = tokens[0]
            wav_path = Path(dataset_dir) / subFolder / 'wav' / wavname[:7] / wavname
            if(wav_path.is_file()):
                wavname = wavname.split('.')[0]
                pinyin = []
                for token in tokens[2::2]:
                    pinyin.append(token)

                pinyin = ' '.join(pinyin)
                text_dict[wavname] = (wav_path, 'aishell3', wavname[:7], pinyin)

    return text_dict


def aishell(dataset_dir: str) -> dict:
    text_dict = {}

    label_path = Path(dataset_dir) / 'transcript' / 'aishell_transcript_v0.8.txt'
    labelfile = open(str(label_path), encoding='utf-8')

    all_wavs = get_files(Path(dataset_dir) / 'wav', extension='.wav')
    wav_paths = {w.stem: w for w in all_wavs}

    while(True):
        line = labelfile.readline().strip()
        if(line==None or len(line)<1):
            break

        tokens = line.split()
        name = tokens[0]
        if(name in wav_paths.keys()):
            text = ' '.join(tokens[1:])
            text_dict[name] = (wav_paths[name], 'aishell', name[6:11], text)

    return text_dict


def post_processed_reader(metadata_path: str, column_sep='|', upsample_indicators='?!', upsample_factor=10) -> Tuple[
    Dict, List]:
    """
    Used to read metadata files created within the repo.
    """
    text_dict = {}
    upsample = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            l_split = l.split(column_sep)
            filename, text = l_split[0], l_split[1]
            text = text.replace('\n', '')
            if any(el in text for el in list(upsample_indicators)):
                upsample.extend([filename] * upsample_factor)
            text_dict.update({filename: text})
    return text_dict, upsample


if __name__ == '__main__':
    metadata_path = '/Volumes/data/datasets/LJSpeech-1.1/metadata.csv'
    d = get_preprocessor_by_name('ljspeech')(metadata_path)
    key_list = list(d.keys())
    print('metadata head')
    for key in key_list[:5]:
        print(f'{key}: {d[key]}')
    print('metadata tail')
    for key in key_list[-5:]:
        print(f'{key}: {d[key]}')
