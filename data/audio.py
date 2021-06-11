import sys
import struct

import librosa
import numpy as np
import librosa.display
from matplotlib import pyplot as plt
import soundfile as sf
import webrtcvad
from scipy.ndimage import binary_dilation
import pyworld as pw


class Audio():
    def __init__(self, config: dict):
        self.config = config
        self.normalizer = getattr(sys.modules[__name__], config['normalizer'])()
    
    def _normalize(self, S):
        return self.normalizer.normalize(S)
    
    def _denormalize(self, S):
        return self.normalizer.denormalize(S)
    
    def _linear_to_mel(self, spectrogram):
        return librosa.feature.melspectrogram(
            S=spectrogram,
            sr=self.config['sampling_rate'],
            n_fft=self.config['n_fft'],
            n_mels=self.config['mel_channels'],
            fmin=self.config['f_min'],
            fmax=self.config['f_max'])
    
    def _stft(self, y):
        return librosa.stft(
            y=y,
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length'],
            win_length=self.config['win_length'])
    
    def mel_spectrogram(self, wav):
        """ This is what the model is trained to reproduce. """
        D = self._stft(wav)
        S = self._linear_to_mel(np.abs(D))
        return self._normalize(S).T
    
    def reconstruct_waveform(self, mel, n_iter=32):
        """ Uses Griffin-Lim phase reconstruction to convert from a normalized
        mel spectrogram back into a waveform. """
        amp_mel = self._denormalize(mel)
        S = librosa.feature.inverse.mel_to_stft(
            amp_mel,
            power=1,
            sr=self.config['sampling_rate'],
            n_fft=self.config['n_fft'],
            fmin=self.config['f_min'],
            fmax=self.config['f_max'])
        wav = librosa.core.griffinlim(
            S,
            n_iter=n_iter,
            hop_length=self.config['hop_length'],
            win_length=self.config['win_length'])
        return wav
    
    def display_mel(self, mel, is_normal=True):
        if is_normal:
            mel = self._denormalize(mel)
        f = plt.figure(figsize=(10, 4))
        s_db = librosa.power_to_db(mel, ref=np.max)
        ax = librosa.display.specshow(s_db,
                                      x_axis='time',
                                      y_axis='mel',
                                      sr=self.config['sampling_rate'],
                                      fmin=self.config['f_min'],
                                      fmax=self.config['f_max'])
        f.add_subplot(ax)
        return f
    
    def load_file(self, path, trim_center_vad=False):
        y, sr = librosa.load(path, sr=self.config['sampling_rate'])
        if self.config['trim_long_silences']:
            y = self.trim_long_silences(y)
        if self.config['trim_silence']:
            y = self.trim_silence(y, trim_center_vad)

        if y.shape[0] % self.config['hop_length'] == 0:
            y = np.pad(y, (0, 1))
        return y, sr
    
    def save_wav(self, y, wav_path):
        sf.write(wav_path, data=y, samplerate=self.config['sampling_rate'])
    
    def extract_pitch(self, y):
        _f0, t = pw.dio(y.astype(np.float64), fs=self.config['sampling_rate'],
                        frame_period=self.config['hop_length'] / self.config['sampling_rate'] * 1000)
        f0 = pw.stonemask(y.astype(np.float64), _f0, t, fs=self.config['sampling_rate'])  # pitch refinement
        
        return f0
    
    def trim_silence(self, wav, trim_center_vad):
        if(trim_center_vad):
            slices = librosa.effects.split(wav, top_db=self.config['trim_silence_top_db'])
            trimcenter = []
            for slice in slices:
                trim = wav[slice[0]:slice[1]]
                trimcenter.append(trim)
            trimmed = np.concatenate(trimcenter, axis=-1)
        else:
            trimmed = librosa.effects.trim(wav,
                                        top_db=self.config['trim_silence_top_db'],
                                        frame_length=256,
                                        hop_length=64)[0]

        return trimmed
    
    # from https://github.com/resemble-ai/Resemblyzer/blob/master/resemblyzer/audio.py
    def trim_long_silences(self, wav):
        int16_max = (2 ** 15) - 1
        samples_per_window = (self.config['vad_window_length'] * self.config['vad_sample_rate']) // 1000
        wav = wav[:len(wav) - (len(wav) % samples_per_window)]
        pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
        voice_flags = []
        vad = webrtcvad.Vad(mode=3)
        for window_start in range(0, len(wav), samples_per_window):
            window_end = window_start + samples_per_window
            voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                             sample_rate=self.config['vad_sample_rate']))
        voice_flags = np.array(voice_flags)
        
        def moving_average(array, width):
            array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
            ret = np.cumsum(array_padded, dtype=float)
            ret[width:] = ret[width:] - ret[:-width]
            return ret[width - 1:] / width
        
        audio_mask = moving_average(voice_flags, self.config['vad_moving_average_width'])
        audio_mask = np.round(audio_mask).astype(np.bool)
        audio_mask[:] = binary_dilation(audio_mask[:], np.ones(self.config['vad_max_silence_length'] + 1))
        audio_mask = np.repeat(audio_mask, samples_per_window)
        return wav[audio_mask]


class Normalizer:
    def normalize(self, S):
        raise NotImplementedError
    
    def denormalize(self, S):
        raise NotImplementedError


class MelGAN(Normalizer):
    def __init__(self):
        super().__init__()
        self.clip_min = 1.0e-5
    
    def normalize(self, S):
        S = np.clip(S, a_min=self.clip_min, a_max=None)
        return np.log(S)
    
    def denormalize(self, S):
        return np.exp(S)


class WaveRNN(Normalizer):
    def __init__(self):
        super().__init__()
        self.min_level_db = - 100
        self.max_norm = 4
    
    def normalize(self, S):
        S = self.amp_to_db(S)
        S = np.clip((S - self.min_level_db) / -self.min_level_db, 0, 1)
        return (S * 2 * self.max_norm) - self.max_norm
    
    def denormalize(self, S):
        S = (S + self.max_norm) / (2 * self.max_norm)
        S = (np.clip(S, 0, 1) * -self.min_level_db) + self.min_level_db
        return self.db_to_amp(S)
    
    def amp_to_db(self, x):
        return 20 * np.log10(np.maximum(1e-5, x))
    
    def db_to_amp(self, x):
        return np.power(10.0, x * 0.05)
