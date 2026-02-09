from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
from typing import Optional
import shutil
from pathlib import Path

from audio_analyzer import AudioAnalyzer
from video_generator import VideoGenerator
from glitch_pipeline import GlitchPipeline

app = FastAPI(title="MusicVideo Generator API", version="1.0.0")

# CORS für Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
TEMP_DIR = Path("temp")

for dir_path in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

# Initialize services
audio_analyzer = AudioAnalyzer()
video_generator = VideoGenerator()
glitch_pipeline = GlitchPipeline()

@app.get("/")
async def root():
    return {"message": "🎵 Pixel-Echo MusicVideo Generator API 🎬"}

@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    format: str = Form("16:9"),  # 16:9 oder 9:16
    style: str = Form("surreal")  # surreal, glitch, abstract
):
    """
    Upload Bild + Audio und starte Video-Generierung
    """
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    
    try:
        # Save uploaded files
        image_path = job_dir / f"image.{image.filename.split('.')[-1]}"
        audio_path = job_dir / f"audio.{audio.filename.split('.')[-1]}"
        
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        # Start background processing
        background_tasks.add_task(
            process_video, 
            job_id, 
            str(image_path), 
            str(audio_path), 
            format, 
            style
        )
        
        return JSONResponse({
            "job_id": job_id,
            "status": "processing",
            "message": "Video generation started"
        })
        
    except Exception as e:
        return JSONResponse({
            "error": str(e)
        }, status_code=500)

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Check processing status
    """
    job_dir = TEMP_DIR / job_id
    status_file = job_dir / "status.json"
    
    if not status_file.exists():
        return JSONResponse({
            "error": "Job not found"
        }, status_code=404)
    
    import json
    with open(status_file, "r") as f:
        status = json.load(f)
    
    return status

@app.get("/download/{job_id}")
async def download_video(job_id: str):
    """
    Download generated video
    """
    output_file = OUTPUT_DIR / f"{job_id}.mp4"
    
    if not output_file.exists():
        return JSONResponse({
            "error": "Video not found"
        }, status_code=404)
    
    from fastapi.responses import FileResponse
    return FileResponse(
        output_file,
        media_type="video/mp4",
        filename=f"pixel-echo-video_{job_id}.mp4"
    )

async def process_video(job_id: str, image_path: str, audio_path: str, format: str, style: str):
    """
    Background processing pipeline
    """
    job_dir = TEMP_DIR / job_id
    status_file = job_dir / "status.json"
    
    def update_status(step: str, progress: float, message: str = ""):
        import json
        status = {
            "job_id": job_id,
            "step": step,
            "progress": progress,
            "message": message,
"status": "processing" if progress < 100 else "completed"
        }
        with open(status_file, "w") as f:
            json.dump(status, f)
    
    try:
        # Step 1: Audio Analysis
        update_status("audio_analysis", 10, "Analyzing beats...")
        audio_features = await audio_analyzer.analyze(audio_path)
        
        # Step 2: Generate Glitch Sequence
        update_status("glitch_generation", 30, "Creating glitch effects...")
        glitch_frames = await glitch_pipeline.generate_sequence(
            image_path, audio_features, style, format
        )
        
        # Step 3: Video Assembly
        update_status("video_assembly", 70, "Assembling video...")
        output_path = OUTPUT_DIR / f"{job_id}.mp4"
        await video_generator.create_video(
            glitch_frames, audio_path, str(output_path), format
        )
        
        # Step 4: Complete
        update_status("completed", 100, "🎵 Pixel-Echo Video ready! 🎬")
        
        # Cleanup temp files
        shutil.rmtree(job_dir)
        
    except Exception as e:
        update_status("error", 0, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
