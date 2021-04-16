import argparse
from pathlib import Path
import pickle
import os
from re import S
import librosa
import tensorflow as tf
from tensorflow.core.framework.graph_pb2 import GraphDef
import numpy as np
from tqdm import trange
from concurrent.futures import ThreadPoolExecutor

from utils.logging_utils import SummaryManager
from data.text import TextToTokens
from data.datasets import DataReader
from utils.config_manager import Config
from data.audio import Audio

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/session_paths.yaml')
parser.add_argument('--skip_phonemes', action='store_true')
parser.add_argument('--skip_mels', action='store_true')
parser.add_argument('--skip_speakers', action='store_true')

args = parser.parse_args()
for arg in vars(args):
    print('{}: {}'.format(arg, getattr(args, arg)))

cm = Config(args.config, aligner=True)
cm.create_remove_dirs()
metadatareader = DataReader.from_config(cm, kind='original')
summary_manager = SummaryManager(model=None, log_dir=cm.log_dir / 'data_preprocessing', config=cm.config,
                                 default_writer='data_preprocessing')
print(f'\nFound {len(metadatareader.filenames)} wavfiles.')
audio = Audio(config=cm.config)

if not args.skip_mels:

    len_dict = {}
    remove_files = []
    mel_lens = []

    def process_wav(wav_tuples):
        for idx in trange(len(wav_tuples), desc=''):
            fullpath, data_type, spk_name, _ = wav_tuples[idx]

            file_name = fullpath.stem
            _, trim_type = cm.data_type[data_type]
            try:
                y, sr = audio.load_wav(str(fullpath), trim_center_vad=trim_type=='trimcentervad')
            except Exception as e:
                remove_files.append(file_name)
                continue
            mel = audio.mel_spectrogram(y)
            assert mel.shape[1] == audio.config['mel_channels'], len(mel.shape) == 2

            len_dict.update({file_name: mel.shape[0]})
            if mel.shape[0] > cm.config['max_mel_len'] or mel.shape[0] < cm.config['min_mel_len']:
                remove_files.append(file_name)
            else:
                mel_lens.append(mel.shape[0])
                os.makedirs(str(cm.mel_dir / spk_name), exist_ok=True)
                mel_path = (cm.mel_dir / spk_name / file_name).with_suffix('.npy')
                np.save(str(mel_path), mel)
    
    print(f"\nMels will be stored stored under")
    print(f"{cm.mel_dir}")
    
    wav_tuples = list(metadatareader.text_dict.values())
    len_dict = {}
    remove_files = []
    mel_lens = []

    poolsize = 4
    piecesize = len(wav_tuples)//poolsize
    wav_inputs = []

    for i in range(poolsize):
        wav_inputs.append(wav_tuples[piecesize*i:piecesize*(i+1)])
    wav_inputs[-1].extend(wav_tuples[piecesize*(i+1):])

    with ThreadPoolExecutor() as p:
        p.map(process_wav, wav_inputs)

    pickle.dump(len_dict, open(cm.data_dir / 'mel_len.pkl', 'wb'))
    pickle.dump(remove_files, open(cm.data_dir / 'under-over_sized_mels.pkl', 'wb'))
    summary_manager.add_histogram('Mel Lengths', values=np.array(mel_lens))
    total_mel_len = np.sum(mel_lens)
    total_wav_len = total_mel_len * audio.config['hop_length']
    summary_manager.display_scalar('Total duration (hours)',
                                   scalar_value=total_wav_len / audio.config['sampling_rate'] / 60. ** 2)


if not args.skip_phonemes:
    remove_files = pickle.load(open(cm.data_dir / 'under-over_sized_mels.pkl', 'rb'))
    phonemized_metadata_path = cm.phonemized_metadata_path
    train_metadata_path = cm.train_metadata_path
    test_metadata_path = cm.valid_metadata_path
    print(f'\nFound {len(metadatareader.filenames)} lines.')
    filter_metadata = []
    wav_tuples = metadatareader.text_dict.values()
    for wav_tuple in wav_tuples:
        fullpath, _, _, text = wav_tuple
        if len(text) < 3:
            filter_metadata.append(fullpath.stem)
    if len(filter_metadata) > 0:
        print(f'Removing {len(filter_metadata)} suspiciously short line(s):')
        for fname in filter_metadata:
            print(f'{fname}: {metadatareader.text_dict[fname][-1]}')
    print(f'\nRemoving {len(remove_files)} line(s) due to mel filtering.')
    remove_files += filter_metadata
    metadata_file_ids = [fname for fname in metadatareader.text_dict.keys() if fname not in remove_files]
    metadata_len = len(metadata_file_ids)
    sample_items = np.random.choice(metadata_file_ids, 20)
    test_len = cm.config['n_test']
    train_len = metadata_len - test_len
    print(f'\nMetadata contains {metadata_len} lines.')
    print(f'\nFiles will be stored under {cm.data_dir}')
    print(f' - all: {phonemized_metadata_path}')
    print(f' - {train_len} training lines: {train_metadata_path}')
    print(f' - {test_len} validation lines: {test_metadata_path}')
    
    print('\nMetadata samples:')
    for i in sample_items:
        print(f'{i}:{metadatareader.text_dict[i][-1]}')
        summary_manager.add_text(f'{i}/text', text=metadatareader.text_dict[i][-1])
    
    # run cleaner on raw text
    text_proc = TextToTokens.default(add_start_end=False,
                                     with_stress=cm.config['with_stress'],
                                     njobs=1)
    
    def process_phonemes(file_ids):
        phonemized_dict = {}
        for idx in trange(len(file_ids), desc=''):
            file_id = file_ids[idx]
            _, data_type, speaker, text = metadatareader.text_dict[file_id]

            try:
                language, _ = cm.data_type[data_type]
                phonemes = text_proc.phonemizer(text, language=language)
            except Exception as e:
                print(f'{e}\nFile id {file_id}')
                continue

            phonemized_dict.update({file_id: '|'.join([speaker, phonemes])})
        return phonemized_dict
    
    print('\nPHONEMIZING')
    phonemized_data = {}

    poolsize = 4
    piecesize = len(metadata_file_ids)//poolsize
    metadata_file_inputs = []

    for i in range(poolsize):
        metadata_file_inputs.append(metadata_file_ids[piecesize*i:piecesize*(i+1)])
    metadata_file_inputs[-1].extend(metadata_file_ids[piecesize*(i+1):])

    with ThreadPoolExecutor() as p:
        procs = p.map(process_phonemes, metadata_file_inputs)
        for proc in procs:
            phonemized_data.update(proc)

    print('\nPhonemized metadata samples:')
    for i in sample_items:
        print(f'{i}:{phonemized_data[i]}')
        summary_manager.add_text(f'{i}/phonemes', text=phonemized_data[i])
    
    new_metadata = [f'{k}|{v}\n' for k, v in phonemized_data.items()]
    shuffled_metadata = np.random.permutation(new_metadata)
    train_metadata = shuffled_metadata[0:train_len]
    test_metadata = shuffled_metadata[-test_len:]
    
    with open(phonemized_metadata_path, 'w+', encoding='utf-8') as file:
        file.writelines(new_metadata)
    with open(train_metadata_path, 'w+', encoding='utf-8') as file:
        file.writelines(train_metadata)
    with open(test_metadata_path, 'w+', encoding='utf-8') as file:
        file.writelines(test_metadata)
    # some checks
    assert metadata_len == len(set(list(phonemized_data.keys()))), \
        f'Length of metadata ({metadata_len}) does not match the length of the phoneme array ({len(set(list(phonemized_data.keys())))}). Check for empty text lines in metadata.'
    assert len(train_metadata) + len(test_metadata) == metadata_len, \
        f'Train and/or validation lengths incorrect. ({len(train_metadata)} + {len(test_metadata)} != {metadata_len})'

    allphonemes = set()
    for file_id in phonemized_data:
        for phon in list(phonemized_data[file_id]):
            allphonemes.add(phon.split('|')[-1])

    with open('all_phonemes.txt', 'w', encoding='utf-8') as file:
        allphonemes = ''.join(allphonemes)
        file.writelines(allphonemes)


if not args.skip_speakers:

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

    wav_tuples = metadatareader.text_dict.values()
    speaker2wav = {}
    for wav_tuple in wav_tuples:
        fullpath, _, speaker, _ = wav_tuple
        if(speaker in speaker2wav.keys()):
            speaker2wav[speaker].append(fullpath)
        else:
            speaker2wav[speaker] = [fullpath]

    print(f'\nTotal {len(speaker2wav.keys())} speakers.')
    mean_dict = {}
    for idx, speaker in enumerate(speaker2wav):
        print(f"{speaker} --> {idx}:{len(speaker2wav)}")
        spk_embedding = []
        for wav in speaker2wav[speaker][:100]:
            specs = load_data(wav)
            if(specs is None or specs.shape[0]<50):
                continue
            specs = np.expand_dims(np.expand_dims(specs, 0), -1)
            embedding = graphModel(tf.convert_to_tensor(specs, tf.float32))['embeddings'][0]
            spk_embedding.append(embedding)

        median = np.median(spk_embedding, axis=0)
        median /= np.linalg.norm(median)

        distances = []
        for i in range(len(spk_embedding)):
            distance = np.linalg.norm(spk_embedding[i]-median)
            distances.append(distance)

        if(len(distances)<15):
            maxDistance = distances[0]
        else:
            distances = sorted(distances)[::-1]
            maxDistance = distances[14]

        filtered_embeddings = []
        for i in range(len(spk_embedding)):
            distance = np.linalg.norm(spk_embedding[i]-median)
            if(distance<=maxDistance):
                filtered_embeddings.append(spk_embedding[i])

        distances = []
        mean = np.mean(filtered_embeddings, axis=0)
        mean /= np.linalg.norm(mean)
        mean_dict[speaker] = mean

        for i in range(len(filtered_embeddings)):
            distance = np.linalg.norm(filtered_embeddings[i]-mean)
            distances.append(distance)

        print(max(distances), len(filtered_embeddings))

    pickle.dump(mean_dict, open(cm.data_dir / 'spk_emb.pkl', 'wb'))

print('\nDone')
