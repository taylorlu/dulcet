from pathlib import Path
from random import Random
from typing import List, Union

import numpy as np
import tensorflow as tf

from utils.config_manager import Config
from data.text.tokenizer import Tokenizer
from data.metadata_readers import get_preprocessor_by_name, get_files
import pickle


class DataReader:
    """
    Reads dataset folder and constructs three useful objects:
        text_dict: {filename: fullpath, data_type, speaker, text} when create data
                   {filename: speaker, text}                      when training
        filenames: [filename1, filename2, ...]
        
    IMPORTANT: Use only for information available from source dataset, not for
    training data.
    """
    
    def __init__(self, audio_directory: List, metadata_path: str, metadata_reading_function=None, 
                 training=False, is_processed=False):
        self.metadata_reading_function = metadata_reading_function
        self.audio_directory = audio_directory
        if not is_processed:
            self.text_dict = {}
            for dataset in self.audio_directory:
                dataset_dir, data_type = dataset
                reader = get_preprocessor_by_name(data_type)
                self.text_dict.update(reader(dataset_dir))
        else:
            self.text_dict = self.metadata_reading_function(Path(metadata_path))
        self.filenames = list(self.text_dict.keys())

    @classmethod
    def from_config(cls, config_manager: Config, kind: str):
        kinds = ['original', 'phonemized', 'train', 'valid']
        if kind not in kinds:
            raise ValueError(f'Invalid kind type. Expected one of: {kinds}')
        reader = get_preprocessor_by_name('post_processed_reader')
        training = False
        is_processed = True
        if kind == 'train':
            metadata = config_manager.train_metadata_path
            training = True
        elif kind == 'original':
            metadata = '.'
            reader = None
            is_processed = False
        elif kind == 'valid':
            metadata = config_manager.valid_metadata_path
        elif kind == 'phonemized':
            metadata = config_manager.phonemized_metadata_path
        
        return cls(audio_directory=config_manager.audio_directory,
                   metadata_reading_function=reader,
                   metadata_path=metadata,
                   training=training,
                   is_processed=is_processed)


class ASRDataset:
    def __init__(self,
                 data_reader: DataReader,
                 mel_directory: str,
                 mel_channels: int,
                 tokenizer: Tokenizer,
                 spk_dict: dict):
        self.metadata_reader = data_reader
        self.mel_directory = Path(mel_directory)
        self.mel_channels = mel_channels
        self.tokenizer = tokenizer
        self.spk_dict = spk_dict
    
    def _read_sample(self, sample_name: str):
        spk, text = self.metadata_reader.text_dict[sample_name]
        mel = np.load((self.mel_directory / spk / sample_name).with_suffix('.npy').as_posix())
        encoded_phonemes = self.tokenizer(text)
        spk = self.spk_dict[spk]
        return spk, mel, encoded_phonemes, mel.shape[0], len(encoded_phonemes), sample_name
    
    def get_sample_length(self, spk, mel, encoded_phonemes, mel_len, phon_len, sample_name):
        return tf.shape(mel)[0]
    
    def get_dataset(self, bucket_batch_sizes, bucket_boundaries, shuffle=True, drop_remainder=False):
        return Dataset(samples=self.metadata_reader.filenames,
                        preprocessor=self._read_sample,
                        output_types=(tf.int32, tf.float32, tf.int32, tf.int32, tf.int32, tf.string),
                        padded_shapes=([], [None, self.mel_channels], [None], [], [], []),
                        len_function=self.get_sample_length,
                        shuffle=shuffle,
                        drop_remainder=drop_remainder,
                        bucket_batch_sizes=bucket_batch_sizes,
                        bucket_boundaries=bucket_boundaries)
    
    @classmethod
    def from_config(cls,
                    config: Config,
                    kind: str,
                    tokenizer: Tokenizer,
                    mel_directory: str = None):
        kinds = ['phonemized', 'train', 'valid']
        if kind not in kinds:
            raise ValueError(f'Invalid kind type. Expected one of: {kinds}')
        if mel_directory is None:
            mel_directory = config.mel_dir
        metadata_reader = DataReader.from_config(config,
                                                 kind=kind)
        return cls(data_reader=metadata_reader,
                   mel_directory=mel_directory,
                   mel_channels=config.config['mel_channels'],
                   tokenizer=tokenizer,
                   spk_dict=config.spk_dict)


class MelDurDataset:
    def __init__(self,
                 data_reader: DataReader,
                 mel_directory: str,
                 mel_channels: int,
                 tokenizer: Tokenizer,
                 spk_dict: dict):
        self.metadata_reader = data_reader
        self.mel_directory = Path(mel_directory)
        self.mel_channels = mel_channels
        self.tokenizer = tokenizer
        self.spk_dict = spk_dict
    
    def _read_sample(self, sample_name: str):
        spk_name, text = self.metadata_reader.text_dict[sample_name]
        mel = np.load((self.mel_directory / spk_name / sample_name).with_suffix('.npy').as_posix())
        encoded_phonemes = self.tokenizer(text)
        return spk_name, mel, encoded_phonemes, mel.shape[0], len(encoded_phonemes), sample_name
    
    def get_sample_length(self, spk_name, mel, encoded_phonemes, mel_len, phon_len, sample_name):
        return tf.shape(mel)[0]
    
    def get_dataset(self, bucket_batch_sizes, bucket_boundaries, shuffle=True, drop_remainder=False):
        return Dataset(samples=self.metadata_reader.filenames,
                        preprocessor=self._read_sample,
                        output_types=(tf.string, tf.float32, tf.int32, tf.int32, tf.int32, tf.string),
                        padded_shapes=([], [None, self.mel_channels], [None], [], [], []),
                        len_function=self.get_sample_length,
                        shuffle=shuffle,
                        drop_remainder=drop_remainder,
                        bucket_batch_sizes=bucket_batch_sizes,
                        bucket_boundaries=bucket_boundaries)
    
    @classmethod
    def from_config(cls,
                    config: Config,
                    kind: str,
                    tokenizer: Tokenizer,
                    mel_directory: str = None):
        kinds = ['phonemized', 'train', 'valid']
        if kind not in kinds:
            raise ValueError(f'Invalid kind type. Expected one of: {kinds}')
        if mel_directory is None:
            mel_directory = config.mel_dir
        metadata_reader = DataReader.from_config(config,
                                                 kind=kind)
        return cls(data_reader=metadata_reader,
                   mel_directory=mel_directory,
                   mel_channels=config.config['mel_channels'],
                   tokenizer=tokenizer,
                   spk_dict=config.spk_dict)


class TTSPreprocessor:
    def __init__(self, 
                 mel_channels: int, 
                 tokenizer: Tokenizer):
        self.output_types = (tf.int32, tf.float32, tf.int32, tf.float32, tf.string)
        self.padded_shapes = ([None], [None, mel_channels], [None], [512], [])
        self.tokenizer = tokenizer
    
    def __call__(self, text, mel, durations, spk_emb, sample_name):
        encoded_phonemes = self.tokenizer(text)
        return encoded_phonemes, mel, durations, spk_emb, sample_name
    
    def get_sample_length(self, encoded_phonemes, mel, durations, spk_emb, sample_name):
        return tf.shape(mel)[0]
    
    @classmethod
    def from_config(cls, config: Config, tokenizer: Tokenizer):
        return cls(mel_channels=config.config['mel_channels'],
                   tokenizer=tokenizer)


class TTSDataset:
    def __init__(self,
                 data_reader: DataReader,
                 preprocessor: TTSPreprocessor,
                 mel_directory: str,
                 duration_directory: str):
        self.metadata_reader = data_reader
        self.preprocessor = preprocessor
        self.mel_directory = Path(mel_directory)
        self.duration_directory = Path(duration_directory)
        self.spk_emb_dict = pickle.load(open(str(self.mel_directory / '../' / 'spk_emb.pkl'), 'rb'))
    
    def _read_sample(self, sample_name: str):
        spk_name, text = self.metadata_reader.text_dict[sample_name]
        spk_emb = self.spk_emb_dict[spk_name]
        mel = np.load((self.mel_directory / spk_name /sample_name).with_suffix('.npy').as_posix())
        durations = np.load(
            (self.duration_directory / spk_name / sample_name).with_suffix('.npy').as_posix())

        return text, mel, durations, spk_emb, sample_name

    def _process_sample(self, sample_name: str):
        text, mel, durations, spk_emb, sample_name = self._read_sample(sample_name)
        return self.preprocessor(text=text, mel=mel, durations=durations, spk_emb=spk_emb, sample_name=sample_name)
    
    def get_dataset(self, bucket_batch_sizes, bucket_boundaries, shuffle=True, drop_remainder=False):
        return Dataset(
            samples=self.metadata_reader.filenames,
            preprocessor=self._process_sample,
            output_types=self.preprocessor.output_types,
            padded_shapes=self.preprocessor.padded_shapes,
            len_function=self.preprocessor.get_sample_length,
            shuffle=shuffle,
            drop_remainder=drop_remainder,
            bucket_batch_sizes=bucket_batch_sizes,
            bucket_boundaries=bucket_boundaries)
    
    @classmethod
    def from_config(cls,
                    config: Config,
                    preprocessor,
                    kind: str,
                    mel_directory: str = None,
                    duration_directory: str = None):
        kinds = ['phonemized', 'train', 'valid']
        if kind not in kinds:
            raise ValueError(f'Invalid kind type. Expected one of: {kinds}')
        if mel_directory is None:
            mel_directory = config.mel_dir
        if duration_directory is None:
            duration_directory = config.duration_dir
        metadata_reader = DataReader.from_config(config,
                                                 kind=kind)
        return cls(preprocessor=preprocessor,
                   data_reader=metadata_reader,
                   mel_directory=mel_directory,
                   duration_directory=duration_directory)


class Dataset:
    """ Model digestible dataset. """
    
    def __init__(self,
                 samples: list,
                 preprocessor,
                 len_function,
                 padded_shapes: tuple,
                 output_types: tuple,
                 bucket_boundaries: list,
                 bucket_batch_sizes: list,
                 padding_values: tuple = None,
                 shuffle=True,
                 drop_remainder=True,
                 seed=42):
        self._random = Random(seed)
        self._samples = samples[:]
        self.preprocessor = preprocessor
        dataset = tf.data.Dataset.from_generator(lambda: self._datagen(shuffle),
                                                 output_types=output_types)
        # TODO: pass bin args
        binned_data = dataset.apply(
            tf.data.experimental.bucket_by_sequence_length(
                len_function,
                bucket_boundaries=bucket_boundaries,
                bucket_batch_sizes=bucket_batch_sizes,
                padded_shapes=padded_shapes,
                drop_remainder=drop_remainder,
                padding_values=padding_values
            ))
        self.dataset = binned_data
        self.data_iter = iter(binned_data.repeat(-1))
    
    def next_batch(self):
        return next(self.data_iter)
    
    def all_batches(self):
        return iter(self.dataset)
    
    def _datagen(self, shuffle):
        """
        Shuffle once before generating to avoid buffering
        """
        samples = self._samples[:]
        if shuffle:
            self._random.shuffle(samples)
        return (self.preprocessor(s) for s in samples)
