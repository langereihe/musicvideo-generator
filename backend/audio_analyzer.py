import librosa
import numpy as np
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class AudioFeatures:
"""Audio analysis results"""
duration: float
tempo: float
beats: List[float]
beat_frames: List[int]
spectral_centroids: np.ndarray
mfcc: np.ndarray
chroma: np.ndarray
rolloff: np.ndarray
zero_crossing_rate: np.ndarray
energy: np.ndarray
mood: Dict[str, float]

class AudioAnalyzer:
"""Audio analysis for beat detection and mood extraction"""

def __init__(self, sample_rate: int = 22050):
self.sample_rate = sample_rate

async def analyze(self, audio_path: str) -> AudioFeatures:
"""
Complete audio analysis pipeline
"""
print(f"🎵 Loading audio: {audio_path}")

# Load audio
y, sr = librosa.load(audio_path, sr=self.sample_rate)

# Basic info
duration = librosa.get_duration(y=y, sr=sr)

# Beat detection
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')[1]

# Audio features
spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
zcr = librosa.feature.zero_crossing_rate(y)[0]

# Energy/RMS
energy = librosa.feature.rms(y=y)[0]

# Mood analysis
mood = self._analyze_mood(y, sr, spectral_centroids, mfcc, chroma, energy)

print(f"✨ Analysis complete: {len(beats)} beats, {tempo:.1f} BPM, mood: {mood}")

return AudioFeatures(
duration=duration,
tempo=tempo,
beats=beats.tolist(),
beat_frames=beat_frames.tolist(),
spectral_centroids=spectral_centroids,
mfcc=mfcc,
chroma=chroma,
rolloff=rolloff,
zero_crossing_rate=zcr,
energy=energy,
mood=mood
)

def _analyze_mood(self, y: np.ndarray, sr: int, centroids: np.ndarray,
mfcc: np.ndarray, chroma: np.ndarray, energy: np.ndarray) -> Dict[str, float]:
"""
Extract mood features from audio
Based on Spotify Audio Features methodology
"""

# Valence (musical positiveness)
# Higher centroid + major chroma = more positive
valence = np.mean(centroids) / 4000.0 # Normalize
major_weight = np.mean(chroma[[0, 2, 4, 5, 7, 9, 11], :]) # Major scale notes
valence = (valence + major_weight) / 2
valence = np.clip(valence, 0, 1)

# Energy (intensity and power)
energy_score = np.mean(energy)
energy_score = np.clip(energy_score * 10, 0, 1) # Scale to 0-1

# Danceability (based on rhythm regularity)
# More regular beats = higher danceability
tempo = librosa.beat.tempo(y=y, sr=sr)[0]
dance_score = 1.0 if 90 <= tempo <= 140 else 0.5 # Optimal dance tempo
dance_score = np.clip(dance_score, 0, 1)

# Acousticness (spectral characteristics)
# Lower energy in higher frequencies = more acoustic
acousticness = 1.0 - np.mean(centroids) / 4000.0
acousticness = np.clip(acousticness, 0, 1)

# Loudness (RMS energy)
loudness = np.mean(energy)
loudness = np.clip(loudness * 5, 0, 1)

return {
"valence": float(valence), # 0-1: sad to happy
"energy": float(energy_score), # 0-1: calm to energetic
"danceability": float(dance_score), # 0-1: not danceable to very danceable
"acousticness": float(acousticness), # 0-1: electronic to acoustic
"loudness": float(loudness) # 0-1: quiet to loud
}

def get_beat_segments(self, features: AudioFeatures, segment_length: float = 4.0) -> List[Dict]:
"""
Split audio into beat-aligned segments for video sync
"""
segments = []
beats = features.beats

current_time = 0.0
segment_start = 0

for i, beat_time in enumerate(beats):
if beat_time - current_time >= segment_length:
# Create segment
segment_beats = beats[segment_start:i]
segment_duration = beat_time - current_time
segments.append({
"start_time": current_time,
"end_time": beat_time,
"duration": segment_duration,
"beats": segment_beats.tolist() if isinstance(segment_beats, np.ndarray) else segment_beats,
"beat_count": len(segment_beats),
"intensity": self._calculate_segment_intensity(features, current_time, beat_time)
})

current_time = beat_time
segment_start = i

return segments

def _calculate_segment_intensity(self, features: AudioFeatures, start_time: float, end_time: float) -> float:
"""
Calculate intensity for a time segment
"""
# Convert time to frame indices
frames_per_second = len(features.energy) / features.duration
start_frame = int(start_time * frames_per_second)
end_frame = int(end_time * frames_per_second)

# Average energy in segment
segment_energy = features.energy[start_frame:end_frame]
intensity = np.mean(segment_energy) if len(segment_energy) > 0 else 0.5

return float(np.clip(intensity * 5, 0, 1)) # Scale to 0-1
