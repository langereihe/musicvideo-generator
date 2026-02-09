import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import random
from typing import List, Dict, Tuple
from pathlib import Path
import json

from audio_analyzer import AudioFeatures, AudioAnalyzer

class GlitchPipeline:
    """Generate beat-synchronized glitch effects"""
    
    def __init__(self):
        self.effects_library = {
            "pixel_shift": self._pixel_shift,
            "rgb_split": self._rgb_split,
            "scan_lines": self._scan_lines,
            "color_corruption": self._color_corruption,
            "digital_noise": self._digital_noise,
            "wave_distortion": self._wave_distortion,
            "zoom_glitch": self._zoom_glitch,
            "color_inversion": self._color_inversion
        }
    
    async def generate_sequence(self, image_path: str, audio_features: AudioFeatures, 
                              style: str, format: str) -> List[str]:
        """
        Generate complete glitch sequence synchronized to beats
        """
        print(f"🎨 Generating glitch sequence for {image_path}")
        
        # Load base image
        base_image = cv2.imread(image_path)
        if base_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize to target format
        base_image = self._resize_to_format(base_image, format)
        
        # Get audio analyzer for beat segments
        analyzer = AudioAnalyzer()
        beat_segments = analyzer.get_beat_segments(audio_features, segment_length=2.0)
        
        # Generate frame sequence
        frame_paths = []
        frames_per_second = 30
        
        print(f"✨ Generating {len(beat_segments)} segments...")
        
        for i, segment in enumerate(beat_segments):
            segment_frames = int(segment["duration"] * frames_per_second)
            intensity = segment["intensity"]
            
            # Choose effects based on mood and intensity
            effects = self._select_effects(audio_features.mood, intensity, style)
            
            print(f"🎬 Segment {i+1}/{len(beat_segments)}: {segment_frames} frames, intensity: {intensity:.2f}")
            
            # Generate frames for this segment
            for frame_idx in range(segment_frames):
                frame_progress = frame_idx / max(segment_frames - 1, 1)
                
                # Apply effects with beat sync
                glitched_frame = self._apply_effects_sequence(
                    base_image.copy(), effects, intensity, frame_progress, segment["beats"]
                )
                
                # Save frame
                frame_path = f"temp/frame_{len(frame_paths):06d}.jpg"
                cv2.imwrite(frame_path, glitched_frame)
                frame_paths.append(frame_path)
        
        print(f"🎉 Generated {len(frame_paths)} frames")
        return frame_paths
    
    def _resize_to_format(self, image: np.ndarray, format: str) -> np.ndarray:
        """Resize image to target aspect ratio"""
        height, width = image.shape[:2]
        
        if format == "16:9":
            target_width, target_height = 1920, 1080
        elif format == "9:16":
            target_width, target_height = 1080, 1920
        else:
            return image
        
        # Maintain aspect ratio while fitting
        scale_w = target_width / width
        scale_h = target_height / height
        scale = min(scale_w, scale_h)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Pad to exact dimensions
        result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return result
    
    def _select_effects(self, mood: Dict[str, float], intensity: float, style: str) -> List[str]:
        """Select effects based on audio mood and style"""
        effects = []
        
        if style == "surreal":
            base_effects = ["wave_distortion", "color_corruption"]
            if mood["energy"] > 0.6:
                base_effects.extend(["zoom_glitch", "rgb_split"])
            if mood["valence"] < 0.4:  # Darker mood
                base_effects.extend(["color_inversion", "scan_lines"])
        
        elif style == "glitch":
            base_effects = ["pixel_shift", "rgb_split", "digital_noise"]
            if intensity > 0.7:
                base_effects.extend(["scan_lines", "color_corruption"])
        
        else:  # abstract
            base_effects = ["wave_distortion", "color_corruption"]
            if mood["danceability"] > 0.6:
                base_effects.extend(["zoom_glitch", "rgb_split"])
        
        # Select 2-4 effects based on intensity
        effect_count = max(2, int(intensity * 4))
        effects = random.sample(base_effects, min(effect_count, len(base_effects)))
        
        return effects
    
    def _apply_effects_sequence(self, image: np.ndarray, effects: List[str], 
                              intensity: float, progress: float, beats: List[float]) -> np.ndarray:
        """Apply effects with beat synchronization"""
        result = image.copy()
        
        # Beat synchronization factor
        beat_factor = self._get_beat_factor(progress, beats, intensity)
        
        for effect_name in effects:
            if effect_name in self.effects_library:
                effect_func = self.effects_library[effect_name]
                effect_strength = intensity * beat_factor
                result = effect_func(result, effect_strength)
        
        return result
    
    def _get_beat_factor(self, progress: float, beats: List[float], base_intensity: float) -> float:
        """Calculate beat-synchronized intensity factor"""
        if not beats:
            return 1.0
        
        # Create beat pulse effect
        current_time = progress * 2.0
        beat_distances = [abs(beat - current_time) for beat in beats]
        min_distance = min(beat_distances) if beat_distances else 1.0
        
        # Stronger near beats
        beat_pulse = max(0.3, 1.0 - min_distance * 2)
        noise = random.uniform(0.8, 1.2)
        
        return beat_pulse * noise
    
    # === GLITCH EFFECTS ===
    
    def _pixel_shift(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Shift pixel blocks horizontally"""
        result = image.copy()
        h, w = image.shape[:2]
        
        shift_amount = int(w * strength * 0.1)
        if shift_amount == 0:
            return result
        
        # Random horizontal shifts
        for _ in range(int(10 * strength)):
            y = random.randint(0, h - 20)
            x_start = random.randint(0, max(1, w - shift_amount))
            block_height = random.randint(5, 30)
            
            # Shift block
            if x_start + shift_amount < w:
                result[y:y+block_height, x_start:-shift_amount] = \
                    result[y:y+block_height, x_start+shift_amount:]
        
        return result
    
    def _rgb_split(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Split RGB channels with offset"""
        result = image.copy()
        h, w = image.shape[:2]
        
        offset = int(w * strength * 0.02)
        if offset == 0:
            return result
        
        b, g, r = cv2.split(result)
        
        # Red shift right
        r_shifted = np.zeros_like(r)
        if offset < w:
            r_shifted[:, offset:] = r[:, :-offset]
# Blue shift left
        b_shifted = np.zeros_like(b)
        if offset < w:
            b_shifted[:, :-offset] = b[:, offset:]
        
        result = cv2.merge([b_shifted, g, r_shifted])
        return result
    
    def _scan_lines(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Add scanning lines effect"""
        result = image.copy()
        h, w = image.shape[:2]
        
        line_spacing = max(2, int(10 / max(strength, 0.1)))
        
        for y in range(0, h, line_spacing):
            if y < h:
                result[y, :] = result[y, :] * (1 - strength * 0.5)
        
        return result
    
    def _color_corruption(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Corrupt color channels randomly"""
        result = image.copy().astype(np.float32)
        
        # Random color shifts
        for channel in range(3):
            shift = random.uniform(-50 * strength, 50 * strength)
            result[:, :, channel] += shift
        
        # Clamp values
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    
    def _digital_noise(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Add digital noise"""
        result = image.copy()
        noise = np.random.randint(-int(50 * strength), int(50 * strength), image.shape, dtype=np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return result
    
    def _wave_distortion(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply wave distortion"""
        h, w = image.shape[:2]
        
        # Create displacement maps
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Wave parameters
        amplitude = strength * 20
        frequency = strength * 0.02
        
        # Calculate displacements
        dx = amplitude * np.sin(frequency * y)
        dy = amplitude * np.cos(frequency * x)
        
        # Apply displacement
        x_new = np.clip(x + dx, 0, w - 1).astype(np.float32)
        y_new = np.clip(y + dy, 0, h - 1).astype(np.float32)
        
        result = cv2.remap(image, x_new, y_new, cv2.INTER_LINEAR)
        return result
    
    def _zoom_glitch(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Random zoom/scale effects"""
        h, w = image.shape[:2]
        
        # Random zoom factor
        zoom = 1.0 + (strength * 0.5 * random.uniform(-1, 1))
        
        # Center point
        cx, cy = w // 2, h // 2
        
        # Create transformation matrix
        M = cv2.getRotationMatrix2D((cx, cy), 0, zoom)
        
        result = cv2.warpAffine(image, M, (w, h))
        return result
    
    def _color_inversion(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Partially invert colors"""
        result = image.copy().astype(np.float32)
        h, w = image.shape[:2]
        
        for _ in range(int(8 * strength)):
            # Random patch
            x1 = random.randint(0, w // 2)
            y1 = random.randint(0, h // 2)
            x2 = min(w, x1 + random.randint(20, w // 4))
            y2 = min(h, y1 + random.randint(20, h // 4))
            
            # Invert patch
            result[y1:y2, x1:x2] = 255 - result[y1:y2, x1:x2]
        
        return result.astype(np.uint8)
