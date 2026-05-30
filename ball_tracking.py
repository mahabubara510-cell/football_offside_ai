import numpy as np
from collections import deque
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch, draw_paths_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer

# ── Config ────────────────────────────────────────────────────────────────────
SOURCE_VIDEO_PATH = r'E:\Computer Vision\Arctech\data\121364_0.mp4'

CONFIG        = SoccerPitchConfiguration()
BALL_ID       = 0
MAXLEN        = 5       # homography smoothing window
RADAR_SCALE   = 0.065
RADAR_PADDING = 20

# ── Models ────────────────────────────────────────────────────────────────────
player_detector = YOLO(r'E:\Computer Vision\Arctech\data\football-player-detection.pt').to('cuda')
pitch_detector  = YOLO(r'E:\Computer Vision\Arctech\data\football-pitch-detection.pt').to('cuda')

# ── Pass 1: collect full ball path ────────────────────────────────────────────
video_info      = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

path_raw = []          # one entry per frame: array of shape (N, 2) or (0, 2)
M        = deque(maxlen=MAXLEN)   # smoothing buffer for homography matrices

for frame in tqdm(frame_generator, total=video_info.total_frames, desc='collecting ball path'):

    # ── Ball detection ────────────────────────────────────────────────────────
    player_result   = player_detector(frame, conf=0.3, device='cuda')[0]
    detections      = sv.Detections.from_ultralytics(player_result)
    ball_detections = detections[detections.class_id == BALL_ID]
    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    # ── Pitch keypoints ───────────────────────────────────────────────────────
    pitch_result   = pitch_detector(frame, conf=0.3, device='cuda')[0]
    key_points     = sv.KeyPoints.from_ultralytics(pitch_result)

    kp_filter              = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][kp_filter]
    pitch_reference_points = np.array(CONFIG.vertices)[kp_filter]

    if len(frame_reference_points) >= 4:
        transformer = ViewTransformer(
            source=frame_reference_points,
            target=pitch_reference_points
        )
        # smooth homography over last MAXLEN frames
        M.append(transformer.m)
        transformer.m = np.mean(np.array(M), axis=0)

        frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)
    else:
        pitch_ball_xy = np.empty((0, 2), dtype=np.float32)

    path_raw.append(pitch_ball_xy)

# ── Build path: keep only frames where exactly 1 ball was detected ────────────
# (drop frames with 0 detections OR ambiguous multiple detections)
path = [
    coords if coords.shape[0] == 1 else np.empty((0, 2), dtype=np.float32)
    for coords in path_raw
]
path = [coords.flatten() for coords in path]   # each entry: (2,) or (0,)

# ── Draw final pitch with ball trail ─────────────────────────────────────────
annotated_frame = draw_pitch(
    CONFIG,
    background_color=sv.Color.from_hex('#1a7a1a'),
    line_color=sv.Color.from_hex('#ffffff'),
    padding=RADAR_PADDING,
    scale=RADAR_SCALE
)
annotated_frame = draw_paths_on_pitch(
    config=CONFIG,
    paths=[path],
    color=sv.Color.WHITE,
    pitch=annotated_frame,
    padding=RADAR_PADDING,
    scale=RADAR_SCALE
)

sv.plot_image(annotated_frame)
print("Ball tracking complete.")