import numpy as np
import librosa
import torch
import torchaudio

class AudioAugmentor:
    def __init__(self,
                 noise_level_impulsive=0.002,
                 noise_level_non_impulsive=0.01,
                 time_stretch_range=(0.8, 1.2),
                 pitch_shift_steps=(-2, 2),
                 volume_gain_db=(-5, 5),
                 clip_threshold=0.5):
        self.noise_level_impulsive = noise_level_impulsive
        self.noise_level_non_impulsive = noise_level_non_impulsive
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_steps = pitch_shift_steps
        self.volume_gain_db = volume_gain_db
        self.clip_threshold = clip_threshold

    def add_noise(self, y, noise_level):
        noise = np.random.randn(len(y))
        return y + noise_level * noise

    def time_stretch(self, y, rate):
        return librosa.effects.time_stretch(y, rate=rate)

    def pitch_shift(self, y, sr, n_steps):
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    def change_volume(self, y, gain_db):
        factor = 10.0 ** (gain_db / 20.0)
        return y * factor

    def clip_audio(self, y, threshold):
        return np.clip(y, -threshold, threshold)

    def augment(self, y, sr, is_impulsive=False):
        if is_impulsive:
            # Safe augmentations only
            y_aug = self.add_noise(y, self.noise_level_impulsive)
            gain_db = np.random.uniform(*self.volume_gain_db)
            y_aug = self.change_volume(y_aug, gain_db)
        else:
            # Non-impulsive: allow full pipeline
            y_aug = self.add_noise(y, self.noise_level_non_impulsive)

            # Time stretch
            stretch_rate = np.random.uniform(*self.time_stretch_range)
            y_aug = self.time_stretch(y_aug, stretch_rate)

            # Pitch shift
            n_steps = np.random.uniform(*self.pitch_shift_steps)
            y_aug = self.pitch_shift(y_aug, sr, n_steps)

            # Volume
            gain_db = np.random.uniform(*self.volume_gain_db)
            y_aug = self.change_volume(y_aug, gain_db)

            # Clipping
            y_aug = self.clip_audio(y_aug, self.clip_threshold)

        return torch.tensor(y_aug).view(1,-1).float()
