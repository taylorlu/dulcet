from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pickle
from pypinyin import lazy_pinyin, Style
from utils.config_manager import Config
from data.audio import Audio
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', dest='config')
    parser.add_argument('--text', '-t', dest='text', default='涉及背景颜色定制，部分画面素材替换', type=str)
    parser.add_argument('--file', '-f', dest='file', default=None, type=str)
    parser.add_argument('--checkpoint', '-ckpt', dest='checkpoint', default=None, type=str)
    parser.add_argument('--outdir', '-o', dest='outdir', default=None, type=str)
    parser.add_argument('--store_mel', '-m', dest='store_mel', action='store_true')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true')
    parser.add_argument('--all_weights', '-ww', dest='all_weights', action='store_true')
    parser.add_argument('--single', '-s', dest='single', action='store_true')
    args = parser.parse_args()
    spk_emb_dict = pickle.load(open(str('/root/mydata/VCTK-Corpus/transformer_tts_data.corpus/spk_emb.pkl'), 'rb'))
    # spk_emb_dict = pickle.load(open(str('test_spk_emb.pkl'), 'rb'))

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
            phons = model.text_pipeline.phonemizer(text_line)
            print(f'Phonemes before: "{phons}"')
            pylist = lazy_pinyin(text_line, style=Style.TONE3)

            tones = ['1','2','3','4','5','ɜ']
            tone_phonemes = []
            for j, phon in enumerate(phons.split(' ')):
                phon = list(phon)
                for i, ch in enumerate(phon):
                    if(ch in tones):
                        phon[i] = list(pylist[j])[-1] if(list(pylist[j])[-1] in tones) else '5'
                tone_phonemes.append(''.join(phon))
            phons = ' '.join(tone_phonemes)
            phons = 'ðiːz fɹˈiː ɐstɹˈɑːlədʒi lˈɛsənz ɑːɹ ɹˈɪʔn fɔːɹ bɪɡˈɪnɚz tə lˈɜːn ɹˈiːəl ɐstɹˈɑːlədʒi? s.ˈo-4 tɕˈi2 pˈei4 tɕˈi3ŋ jˈiɛ2n sˈo-4 tˈi4ŋ ts.ˈi.4 pˈu5 fˈə4n xwˈɑ4 mˈiɛ4n sˈu4 tshˈai4 thˈi2 xˈua4n'

            tokens = model.text_pipeline.tokenizer(phons)
            if args.verbose:
                print(f'Predicting {text_line}')
                print(f'Phonemes: "{phons}"')
                print(f'Tokens: "{tokens}"')
            spk_emb = spk_emb_dict['SSB0375'][np.newaxis, :]
            out = model.predict(tokens, spk_emb=spk_emb, encode=False, phoneme_max_duration=None, speed_regulator=1.0)
            mel = out['mel'].numpy().T
            wav = audio.reconstruct_waveform(mel)
            wavs.append(wav)
            if args.store_mel:
                np.save((outdir / (file_name + f'_{i}')).with_suffix('.mel'), out['mel'].numpy())
            if args.single:
                audio.save_wav(wav, (outdir / (file_name + f'_{i}')).with_suffix('.wav'))
        audio.save_wav(np.concatenate(wavs), (outdir / file_name).with_suffix('.wav'))
