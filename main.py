import os
import uuid
import asyncio
import cv2
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}")

# ── Load models once at startup ───────────────────────────────────────────────
pitch_detector = YOLO(hf_hub_download(
    repo_id="Sabkat/football-pitch-detection",
    filename="football-pitch-detection.pt"
)).to(device)

player_detector = YOLO(hf_hub_download(
    repo_id="Sabkat/football-player-detection",
    filename="football-player-detection.pt"
)).to(device)

# ── Dirs ──────────────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Offside Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/static", StaticFiles(directory="."), name="static")


# ── CPU helper ────────────────────────────────────────────────────────────────
def reduce_video_for_cpu(input_path: str, output_path: str, max_frames: int = 60):
    """
    On CPU deployments: halve the resolution and cap at max_frames.
    This makes YOLO inference feasible without a GPU.
    """
    cap = cv2.VideoCapture(input_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  // 2)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"[CPU mode] Reduced to {frame_count} frames @ {width}x{height}")
    return frame_count


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def serve_index():
    return FileResponse("index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "device": device}


@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):

    # ── 1. Save upload ────────────────────────────────────────────────────────
    job_id     = uuid.uuid4().hex
    ext        = Path(video.filename).suffix or ".mp4"
    input_path = UPLOAD_DIR / f"{job_id}_input{ext}"
    output_path = OUTPUT_DIR / f"{job_id}_output.mp4"

    with open(input_path, "wb") as f:
        content = await video.read()
        f.write(content)

    # ── 2. CPU mode: reduce video before processing ───────────────────────────
    if device == 'cpu':
        reduced_path = UPLOAD_DIR / f"{job_id}_reduced.mp4"
        try:
            frame_count = reduce_video_for_cpu(
                input_path=str(input_path),
                output_path=str(reduced_path),
                max_frames=60
            )
            if frame_count == 0:
                input_path.unlink(missing_ok=True)
                raise HTTPException(status_code=422, detail="Could not read video file.")
            # swap to reduced video for all downstream processing
            input_path.unlink(missing_ok=True)
            input_path = reduced_path
        except HTTPException:
            raise
        except Exception as e:
            input_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Video reduction error: {str(e)}")

    # ── 3. Validate ───────────────────────────────────────────────────────────
    try:
        from validator import validate_video
        loop = asyncio.get_event_loop()
        valid, reason = await loop.run_in_executor(
            None,
            validate_video,
            str(input_path),
            pitch_detector,
            player_detector,
        )
    except Exception as e:
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")

    if not valid:
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=reason)

    # ── 4. Run pipeline ───────────────────────────────────────────────────────
    try:
        from pipeline_3 import run_pipeline
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            run_pipeline,
            str(input_path),
            str(output_path),
        )
    except Exception as e:
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    # ── 5. Cleanup + respond ──────────────────────────────────────────────────
    input_path.unlink(missing_ok=True)

    return JSONResponse({
        "verdict":          result["verdict"],
        "output_video_url": f"/outputs/{output_path.name}",
        "involvement_log":  result["involvement_log"],
        "total_events":     len(result["involvement_log"]),
        "device":           device,
    })
