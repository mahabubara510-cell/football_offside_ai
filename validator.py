import numpy as np
import supervision as sv
from ultralytics import YOLO
from sports.configs.soccer import SoccerPitchConfiguration

MIN_FRAMES          = 30
MIN_PLAYER_CROPS    = 10
MIN_PITCH_FRAMES    = 2
SAMPLE_FRAMES       = 5
CONFIDENCE          = 0.3
KEYPOINT_CONFIDENCE = 0.5
MIN_KEYPOINTS       = 4

CONFIG = SoccerPitchConfiguration()

def validate_video(
    path: str,
    pitch_detector: YOLO,
    player_detector: YOLO,
) -> tuple[bool, str]:
    """
    Pre-flight check before the main pipeline.
    Returns (True, 'ok') or (False, reason_string).
    """

    # ── 1. Readable? ──────────────────────────────────────────────────────────
    try:
        video_info = sv.VideoInfo.from_video_path(path)
    except Exception:
        return False, "Could not read video file. Please upload a valid video."

    # ── 2. Enough frames? ─────────────────────────────────────────────────────
    if video_info.total_frames < MIN_FRAMES:
        return False, f"Video too short ({video_info.total_frames} frames). Minimum is {MIN_FRAMES}."

    # ── 3. Sample frames ──────────────────────────────────────────────────────
    stride = max(1, video_info.total_frames // SAMPLE_FRAMES)
    frame_generator = sv.get_video_frames_generator(path, stride=stride)

    pitch_frames_detected = 0
    total_player_crops    = 0

    for frame in frame_generator:

        # ── Pitch check ───────────────────────────────────────────────────────
        try:
            pitch_result = pitch_detector(frame, conf=CONFIDENCE, verbose=False)[0]
            key_points   = sv.KeyPoints.from_ultralytics(pitch_result)
            kp_filter    = key_points.confidence[0] > KEYPOINT_CONFIDENCE
            if kp_filter.sum() >= MIN_KEYPOINTS:
                pitch_frames_detected += 1
        except Exception:
            pass

        # ── Player crop check ─────────────────────────────────────────────────
        try:
            player_result      = player_detector(frame, conf=CONFIDENCE, verbose=False)[0]
            detections         = sv.Detections.from_ultralytics(player_result)
            players_detections = detections[detections.class_id == 2]  # PLAYER_ID
            total_player_crops += len(players_detections)
        except Exception:
            pass

    # ── 4. Decisions ──────────────────────────────────────────────────────────
    if pitch_frames_detected < MIN_PITCH_FRAMES:
        return False, "No football pitch detected. Please upload a football match video."

    if total_player_crops < MIN_PLAYER_CROPS:
        return False, "Not enough players detected. Please upload a football match video."

    return True, "ok"