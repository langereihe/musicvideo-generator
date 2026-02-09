import cv2
import os
import subprocess
import tempfile
from typing import List
from pathlib import Path

class VideoGenerator:
    """Assemble frames into final video with audio"""
    
    def __init__(self, fps: int = 30):
        self.fps = fps
    
    async def create_video(self, frame_paths: List[str], audio_path: str, 
                         output_path: str, format: str) -> str:
        """
        Create final video from frames and audio
        """
        print(f"🎬 Creating video: {len(frame_paths)} frames → {output_path}")
        
        if not frame_paths:
            raise ValueError("No frames provided")
        
        # Get video dimensions from first frame
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is None:
            raise ValueError(f"Could not read first frame: {frame_paths[0]}")
        
        height, width = first_frame.shape[:2]
        
        # Create temporary video file (without audio)
        temp_video = tempfile.mktemp(suffix='.mp4')
        
        try:
            # Step 1: Create video from frames
            await self._frames_to_video(frame_paths, temp_video, width, height)
            
            # Step 2: Add audio with FFmpeg
            await self._add_audio_ffmpeg(temp_video, audio_path, output_path)
            
            print(f"✅ Pixel-Echo Video created: {output_path}")
            return output_path
            
        finally:
            # Cleanup temporary files
            if os.path.exists(temp_video):
                os.remove(temp_video)
            
            # Cleanup frame files
            for frame_path in frame_paths:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
    
    async def _frames_to_video(self, frame_paths: List[str], output_path: str, 
                             width: int, height: int):
        """Convert frame sequence to video using OpenCV"""
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        try:
            for i, frame_path in enumerate(frame_paths):
                if i % 100 == 0:
                    print(f"🎞️ Processing frame {i+1}/{len(frame_paths)}")
                
                frame = cv2.imread(frame_path)
                if frame is None:
                    print(f"⚠️ Warning: Could not read frame {frame_path}")
                    continue
                
                # Ensure frame dimensions match
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                
                out.write(frame)
                
        finally:
            out.release()
    
    async def _add_audio_ffmpeg(self, video_path: str, audio_path: str, output_path: str):
        """Add audio to video using FFmpeg"""
        
        # FFmpeg command to combine video and audio
        cmd = [
            'ffmpeg', '-y',  # Overwrite output file
            '-i', video_path,  # Input video
            '-i', audio_path,  # Input audio
            '-c:v', 'libx264',  # Video codec
            '-c:a', 'aac',      # Audio codec
            '-strict', 'experimental',
            '-b:a', '192k',     # Audio bitrate
            '-shortest',        # Match shortest stream
            '-movflags', '+faststart',  # Web optimization
            output_path
        ]
        
        print(f"🔊 Running FFmpeg: {' '.join(cmd)}")
        
        # Run FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        print("🎵 Audio synchronized successfully!")
def get_video_info(self, video_path: str) -> dict:
        """Get video information using FFprobe"""
        
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFprobe failed: {result.stderr}")
        
        import json
        return json.loads(result.stdout)
    
    def create_preview_gif(self, frame_paths: List[str], output_path: str, 
                          max_frames: int = 60, scale: float = 0.5) -> str:
        """Create GIF preview from frames"""
        
        # Select subset of frames for GIF
        step = max(1, len(frame_paths) // max_frames)
        preview_frames = frame_paths[::step][:max_frames]
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(preview_frames[0])
        height, width = first_frame.shape[:2]
        
        # Scale down for smaller file size
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Create GIF using imageio (if available) or PIL
        try:
            import imageio
            
            frames = []
            for frame_path in preview_frames:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    # Convert BGR to RGB and resize
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
                    frames.append(frame_resized)
            
            # Save as GIF
            imageio.mimsave(output_path, frames, duration=1/self.fps, loop=0)
            print(f"📸 Preview GIF created: {output_path}")
            
        except ImportError:
            print("⚠️ Warning: imageio not available, skipping GIF creation")
        
        return output_path
