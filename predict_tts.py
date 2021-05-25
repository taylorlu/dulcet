from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pickle
from pypinyin import lazy_pinyin, Style
from utils.config_manager import Config
from data.audio import Audio
import os
import librosa
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def genSpeakerEmbedding(wavpath):
    def load_data(path, sr=16000, win_length=400, hop_length=160, n_fft=512):
        try:
            wav, sr_ret = librosa.load(path, sr=sr)
            intervals = librosa.effects.split(wav, top_db=20)
            wav_output = []
            for sliced in intervals:
                wav_output.extend(wav[sliced[0]:sliced[1]])
            wav = np.array(wav_output)
        except Exception as e:
            print("Exception happened when load_data('{}'): {}".format(path, str(e)))
            return None

        linear_spect = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length).T
        mag, _ = librosa.magphase(linear_spect)  # magnitude
        spec_mag = mag.T

        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)

        return spec_mag

    model = tf.saved_model.load('spkpb')
    graphModel = model.signatures['serving_default']

    specs = load_data(wavpath)
    specs = np.expand_dims(np.expand_dims(specs, 0), -1)
    embedding = graphModel(tf.convert_to_tensor(specs, tf.float32))['embeddings'][0]
    return embedding


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', dest='config', default='config/session_paths.yaml')
    parser.add_argument('--text', '-t', dest='text', default='直接赋值使用的是引用的方式。而有些情况下需要复制整个对象', type=str)
    parser.add_argument('--file', '-f', dest='file', default=None, type=str)
    parser.add_argument('--checkpoint', '-ckpt', dest='checkpoint', default=None, type=str)
    parser.add_argument('--outdir', '-o', dest='outdir', default=None, type=str)
    parser.add_argument('--store_mel', '-m', dest='store_mel', action='store_true')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true')
    parser.add_argument('--all_weights', '-ww', dest='all_weights', action='store_true')
    parser.add_argument('--single', '-s', dest='single', action='store_true')
    args = parser.parse_args()
    # spk_emb_dict = pickle.load(open(str('/root/mydata/Corpus/transformer_tts_data.corpus/spk_emb.pkl'), 'rb'))
    spk_emb_dict = pickle.load(open(str('test_spk_emb.pkl'), 'rb'))
    spk_emb = spk_emb_dict['x_2'][np.newaxis, :]
    spk_emb = genSpeakerEmbedding('ldh.wav')[np.newaxis, :]


    if args.file is not None:
        with open(args.file, 'r') as file:
            text = file.readlines()
        fname = Path(args.file).stem
    elif args.text is not None:
        text = [args.text]
        fname = 'custom_text'
    else:
        fname = None
        text = None
        print(f'Specify either an input text (-t "some text") or a text input file (-f /path/to/file.txt)')
        exit()
    config_loader = Config(config_path=args.config)
    outdir = Path(args.outdir) if args.outdir is not None else config_loader.log_dir
    outdir = outdir / 'outputs' / f'{fname}'
    outdir.mkdir(exist_ok=True, parents=True)
    print('==='*10,outdir)
    audio = Audio(config_loader.config)
    if args.checkpoint is not None:
        all_weights = [args.checkpoint]
    
    elif args.all_weights:
        all_weights = [(config_loader.weights_dir / x.stem).as_posix() for x in config_loader.weights_dir.iterdir() if
                       x.suffix == '.index']
    else:
        all_weights = [None]  # default
    
    if args.verbose:
        print(f'\nWeights list: \n{all_weights}\n')
    for weights in all_weights:
        model = config_loader.load_model(weights)
        file_name = f'{fname}_ttsstep{model.step}'
        print(f'Output wav under {outdir}')
        wavs = []
        for i, text_line in enumerate(text):
            out = model.predict(text_line, spk_emb=spk_emb, encode=True, phoneme_max_duration=None, speed_regulator=0.8)
            mel = out['mel'].numpy().T
            wav = audio.reconstruct_waveform(mel)
            wavs.append(wav)
            if args.store_mel:
                np.save((outdir / (file_name + f'_{i}')).with_suffix('.mel'), out['mel'].numpy())
            if args.single:
                audio.save_wav(wav, (outdir / (file_name + f'_{i}')).with_suffix('.wav'))
        audio.save_wav(np.concatenate(wavs), (outdir / file_name).with_suffix('.wav'))
