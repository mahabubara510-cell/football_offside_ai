import numpy as np
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm
from sports.common.team import TeamClassifier

# ── Models ────────────────────────────────────────────────────────────────────
player_detector = YOLO(r'E:\Computer Vision\Arctech\data\football-player-detection.pt').to('cuda')

# ── Config ────────────────────────────────────────────────────────────────────
SOURCE_VIDEO_PATH = r'E:\Computer Vision\Arctech\data\121364_0.mp4'
TARGET_VIDEO_PATH = r'E:\Computer Vision\Arctech\data\output_players.mp4'

BALL_ID       = 0
GOALKEEPER_ID = 1
PLAYER_ID     = 2
REFEREE_ID    = 3
STRIDE        = 30

# ── Goalkeeper team resolver ───────────────────────────────────────────────────
def resolve_goalkeepers_team_id(
    players: sv.Detections,
    goalkeepers: sv.Detections
) -> np.ndarray:
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy     = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for gk_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(gk_xy - team_0_centroid)
        dist_1 = np.linalg.norm(gk_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)

# ── Step 1: collect crops and fit team classifier ─────────────────────────────
print("Collecting player crops for team classification...")
frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH, stride=STRIDE)

crops = []
for frame in tqdm(frame_generator, desc='collecting crops'):
    result     = player_detector(frame, conf=0.3, device='cuda')[0]
    detections = sv.Detections.from_ultralytics(result)
    players_detections = detections[detections.class_id == PLAYER_ID]
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    crops += players_crops

team_classifier = TeamClassifier(device='cuda')
team_classifier.fit(crops)
print("Team classifier ready.")

# ── Step 2: annotators & tracker ──────────────────────────────────────────────
ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
)
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    text_color=sv.Color.from_hex('#000000'),
    text_position=sv.Position.BOTTOM_CENTER
)
triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FFD700'),
    base=20, height=17
)

tracker = sv.ByteTrack()
tracker.reset()

# ── Step 3: process full video ────────────────────────────────────────────────
video_info      = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    for frame in tqdm(frame_generator, total=video_info.total_frames, desc='processing video'):

        result     = player_detector(frame, conf=0.3)[0]
        detections = sv.Detections.from_ultralytics(result)

        ball_detections      = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        all_detections = detections[detections.class_id != BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections = tracker.update_with_detections(detections=all_detections)

        goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
        players_detections     = all_detections[all_detections.class_id == PLAYER_ID]
        referees_detections    = all_detections[all_detections.class_id == REFEREE_ID]

        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        players_detections.class_id = team_classifier.predict(players_crops)

        if len(goalkeepers_detections) > 0 and len(players_detections) > 0:
            goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
                players_detections, goalkeepers_detections)

        referees_detections.class_id = np.full(len(referees_detections), 2)

        all_detections = sv.Detections.merge([
            players_detections, goalkeepers_detections, referees_detections])
        all_detections.class_id = all_detections.class_id.astype(int)

        labels = [f"#{tid}" for tid in all_detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=all_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=all_detections, labels=labels)
        annotated_frame = triangle_annotator.annotate(scene=annotated_frame, detections=ball_detections)

        sink.write_frame(annotated_frame)

print(f"Done! Saved to {TARGET_VIDEO_PATH}")