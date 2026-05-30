import numpy as np
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer

# ── Models ────────────────────────────────────────────────────────────────────
pitch_detector = YOLO(r'E:\Computer Vision\Arctech\data\football-pitch-detection.pt').to('cuda')

# ── Config ────────────────────────────────────────────────────────────────────
SOURCE_VIDEO_PATH = r'E:\Computer Vision\Arctech\data\121364_0.mp4'
TARGET_VIDEO_PATH = r'E:\Computer Vision\Arctech\data\output_pitch.mp4'

CONFIG = SoccerPitchConfiguration()

# ── Annotators ────────────────────────────────────────────────────────────────
vertex_annotator = sv.VertexAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    radius=8
)
edge_annotator = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#00BFFF'),
    thickness=2,
    edges=CONFIG.edges
)

# ── Process full video ────────────────────────────────────────────────────────
video_info      = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    for frame in tqdm(frame_generator, total=video_info.total_frames, desc='processing pitch'):

        result     = pitch_detector(frame, conf=0.3, device='cuda')[0]
        key_points = sv.KeyPoints.from_ultralytics(result)

        # filter low-confidence keypoints
        filter                  = key_points.confidence[0] > 0.5
        frame_reference_points  = key_points.xy[0][filter]
        pitch_reference_points  = np.array(CONFIG.vertices)[filter]

        annotated_frame = frame.copy()

        if len(frame_reference_points) >= 4:
            # project all pitch vertices onto the frame
            transformer = ViewTransformer(
                source=pitch_reference_points,
                target=frame_reference_points
            )
            pitch_all_points  = np.array(CONFIG.vertices)
            frame_all_points  = transformer.transform_points(points=pitch_all_points)
            frame_all_kp      = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])
            frame_ref_kp      = sv.KeyPoints(xy=frame_reference_points[np.newaxis, ...])

            annotated_frame = edge_annotator.annotate(scene=annotated_frame, key_points=frame_all_kp)
            annotated_frame = vertex_annotator.annotate(scene=annotated_frame, key_points=frame_ref_kp)
        else:
            # fallback: just show detected keypoints
            frame_ref_kp = sv.KeyPoints(xy=frame_reference_points[np.newaxis, ...])
            annotated_frame = vertex_annotator.annotate(scene=annotated_frame, key_points=frame_ref_kp)

        sink.write_frame(annotated_frame)

print(f"Done! Saved to {TARGET_VIDEO_PATH}")