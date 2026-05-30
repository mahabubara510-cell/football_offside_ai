import numpy as np
from collections import deque
import supervision as sv
from tqdm import tqdm
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch, draw_paths_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
from sports.common.team import TeamClassifier
from offside_2 import (
    get_attacking_direction,
    get_offside_line_x,
    check_offside,
    draw_offside_lines_on_radar,
    get_ball_possessor,
    check_offside_involvement,
)
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = SoccerPitchConfiguration()

BALL_ID       = 0
GOALKEEPER_ID = 1
PLAYER_ID     = 2
REFEREE_ID    = 3
STRIDE        = 30

BALL_TRAIL_LEN    = 40
HOMOGRAPHY_SMOOTH = 5

RADAR_SCALE   = 0.065
RADAR_PADDING = 20

# ── Annotators (created once at import time) ──────────────────────────────────
vertex_annotator = sv.VertexAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    radius=8
)
edge_annotator = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#00BFFF'),
    thickness=2,
    edges=CONFIG.edges
)
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def resolve_goalkeepers_team_id(
    players: sv.Detections,
    goalkeepers: sv.Detections
) -> np.ndarray:
    goalkeepers_xy  = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy      = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for gk_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(gk_xy - team_0_centroid)
        dist_1 = np.linalg.norm(gk_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def safe_draw(result, fallback):
    return result if result is not None else fallback


def pad_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h = img.shape[0]
    if h < target_h:
        pad = target_h - h
        img = np.pad(img, ((pad // 2, pad - pad // 2), (0, 0), (0, 0)),
                     mode='constant', constant_values=0)
    return img


# ── Phase 1: collect crops and fit team classifier ────────────────────────────
def collect_crops_and_fit_classifier(
    source_path: str,
    player_detector,
) -> TeamClassifier:
    """
    Samples frames from the video, collects player crops,
    fits and returns a TeamClassifier.
    Models passed in from main.py — not reloaded here.
    """
    print("Collecting player crops for team classification...")
    frame_generator = sv.get_video_frames_generator(
        source_path=source_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result             = player_detector(frame, conf=0.3, device=device)[0]
        detections         = sv.Detections.from_ultralytics(result)
        players_detections = detections[detections.class_id == PLAYER_ID]
        players_crops      = [sv.crop_image(frame, xyxy)
                               for xyxy in players_detections.xyxy]
        crops             += players_crops

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)
    print("Team classifier ready.")
    return team_classifier


# ── Phase 2: full pipeline ────────────────────────────────────────────────────
def run_pipeline(
    source_path: str,
    output_path: str,
    player_detector,
    pitch_detector,
) -> dict:
    """
    Runs the full detection + offside analysis pipeline.
    Models are passed in from main.py — loaded only once across the app.

    Args:
        source_path    : path to the input video
        output_path    : path where the annotated output video will be saved
        player_detector: YOLO model for player/ball/referee detection
        pitch_detector : YOLO model for pitch keypoint detection

    Returns:
        {
            "verdict"         : "OFFSIDE" | "ONSIDE",
            "involvement_log" : list of involvement events,
            "output_path"     : output_path
        }
    """
    # fit classifier fresh for this video
    team_classifier = collect_crops_and_fit_classifier(
        source_path, player_detector)

    tracker = sv.ByteTrack()
    tracker.reset()

    # ── Persistent state ──────────────────────────────────────────────────────
    ball_trail          = deque(maxlen=BALL_TRAIL_LEN)
    homography_M        = deque(maxlen=HOMOGRAPHY_SMOOTH)
    offside_tracker_ids = set()
    previous_possessor  = None
    involvement_log     = []
    frame_number        = 0

    # ── Precompute radar size ─────────────────────────────────────────────────
    _sample_radar = draw_pitch(
        CONFIG,
        background_color=sv.Color.from_hex('#1a7a1a'),
        line_color=sv.Color.from_hex('#ffffff'),
        padding=RADAR_PADDING,
        scale=RADAR_SCALE
    )
    RADAR_H, RADAR_W = _sample_radar.shape[:2]
    print(f"Radar size: {RADAR_W}x{RADAR_H}px")

    video_info      = sv.VideoInfo.from_video_path(source_path)
    frame_generator = sv.get_video_frames_generator(source_path)

    out_height = max(video_info.height, RADAR_H)
    out_width  = video_info.width + RADAR_W
    out_info   = sv.VideoInfo(
        width=out_width,
        height=out_height,
        fps=video_info.fps,
        total_frames=video_info.total_frames
    )

    with sv.VideoSink(output_path, out_info) as sink:
        for frame in tqdm(frame_generator,
                          total=video_info.total_frames,
                          desc='main pipeline'):

            annotated_frame = frame.copy()

            # ── Pitch detection ───────────────────────────────────────────────
            pitch_result = pitch_detector(frame, conf=0.3, device=device)[0]
            key_points   = sv.KeyPoints.from_ultralytics(pitch_result)

            kp_filter              = key_points.confidence[0] > 0.5
            frame_reference_points = key_points.xy[0][kp_filter]
            pitch_reference_points = np.array(CONFIG.vertices)[kp_filter]

            transformer = None

            if len(frame_reference_points) >= 4:
                transformer = ViewTransformer(
                    source=frame_reference_points,
                    target=pitch_reference_points
                )
                homography_M.append(transformer.m)
                transformer.m = np.mean(np.array(homography_M), axis=0)

                overlay_transformer = ViewTransformer(
                    source=pitch_reference_points,
                    target=frame_reference_points
                )
                pitch_all_points = np.array(CONFIG.vertices)
                frame_all_points = overlay_transformer.transform_points(
                    points=pitch_all_points)
                frame_all_kp = sv.KeyPoints(
                    xy=frame_all_points[np.newaxis, ...])
                frame_ref_kp = sv.KeyPoints(
                    xy=frame_reference_points[np.newaxis, ...])

                annotated_frame = edge_annotator.annotate(
                    scene=annotated_frame, key_points=frame_all_kp)
                annotated_frame = vertex_annotator.annotate(
                    scene=annotated_frame, key_points=frame_ref_kp)
            else:
                if len(frame_reference_points) > 0:
                    frame_ref_kp = sv.KeyPoints(
                        xy=frame_reference_points[np.newaxis, ...])
                    annotated_frame = vertex_annotator.annotate(
                        scene=annotated_frame, key_points=frame_ref_kp)

            # ── Player detection ──────────────────────────────────────────────
            player_result = player_detector(frame, conf=0.3, device=device)[0]
            detections    = sv.Detections.from_ultralytics(player_result)

            ball_detections      = detections[detections.class_id == BALL_ID]
            ball_detections.xyxy = sv.pad_boxes(
                xyxy=ball_detections.xyxy, px=10)

            all_detections = detections[detections.class_id != BALL_ID]
            all_detections = all_detections.with_nms(
                threshold=0.5, class_agnostic=True)
            all_detections = tracker.update_with_detections(
                detections=all_detections)

            goalkeepers_detections = all_detections[
                all_detections.class_id == GOALKEEPER_ID]
            players_detections     = all_detections[
                all_detections.class_id == PLAYER_ID]
            referees_detections    = all_detections[
                all_detections.class_id == REFEREE_ID]

            players_crops = [sv.crop_image(frame, xyxy)
                             for xyxy in players_detections.xyxy]
            players_detections.class_id = team_classifier.predict(
                players_crops)

            if (len(goalkeepers_detections) > 0
                    and len(players_detections) > 0):
                goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
                    players_detections, goalkeepers_detections)

            referees_detections.class_id = np.full(
                len(referees_detections), 2)

            all_detections = sv.Detections.merge([
                players_detections,
                goalkeepers_detections,
                referees_detections,
            ])
            all_detections.class_id = all_detections.class_id.astype(int)

            labels = [f"#{tid}" for tid in all_detections.tracker_id]

            annotated_frame = ellipse_annotator.annotate(
                scene=annotated_frame, detections=all_detections)
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=all_detections,
                labels=labels)
            annotated_frame = triangle_annotator.annotate(
                scene=annotated_frame, detections=ball_detections)

            # ── Ball pitch coordinates ────────────────────────────────────────
            ball_pitch_x = None
            ball_pitch_y = None
            if transformer is not None and len(ball_detections) > 0:
                ball_xy       = ball_detections.get_anchors_coordinates(
                    sv.Position.CENTER)
                ball_pitch_xy = transformer.transform_points(points=ball_xy)
                if ball_pitch_xy.shape[0] == 1:
                    ball_pitch_x = float(ball_pitch_xy[0, 0])
                    ball_pitch_y = float(ball_pitch_xy[0, 1])

            # ── Offside logic ─────────────────────────────────────────────────
            direction_info = get_attacking_direction(
                goalkeepers_detections, transformer, CONFIG)

            offside_line_x = get_offside_line_x(
                players_detections, goalkeepers_detections,
                transformer, direction_info, CONFIG)

            offside_mask, offside_tracker_ids = check_offside(
                players_detections, transformer,
                direction_info, offside_line_x, ball_pitch_x,
                offside_tracker_ids
            )

            # ── Possession & involvement ──────────────────────────────────────
            current_possessor = get_ball_possessor(
                players_detections, goalkeepers_detections,
                transformer, ball_pitch_x, ball_pitch_y
            )

            involved_this_frame, involvement_log = check_offside_involvement(
                current_possessor, previous_possessor,
                offside_tracker_ids, involvement_log, frame_number
            )

            previous_possessor = current_possessor
            frame_number += 1

            # ── 2D Radar ──────────────────────────────────────────────────────
            blank = np.zeros((RADAR_H, RADAR_W, 3), dtype=np.uint8)

            radar_frame = safe_draw(draw_pitch(
                CONFIG,
                background_color=sv.Color.from_hex('#1a7a1a'),
                line_color=sv.Color.from_hex('#ffffff'),
                padding=RADAR_PADDING,
                scale=RADAR_SCALE
            ), blank)

            if transformer is not None:

                if len(players_detections) > 0:
                    players_xy = players_detections.get_anchors_coordinates(
                        sv.Position.BOTTOM_CENTER)
                    players_pitch_xy = transformer.transform_points(
                        points=players_xy)

                    team0_mask = players_detections.class_id == 0
                    if team0_mask.any():
                        radar_frame = safe_draw(draw_points_on_pitch(
                            CONFIG,
                            xy=players_pitch_xy[team0_mask],
                            face_color=sv.Color.from_hex('#00BFFF'),
                            edge_color=sv.Color.from_hex('#ffffff'),
                            radius=8,
                            padding=RADAR_PADDING,
                            scale=RADAR_SCALE,
                            pitch=radar_frame
                        ), radar_frame)

                    team1_mask = players_detections.class_id == 1
                    if team1_mask.any():
                        radar_frame = safe_draw(draw_points_on_pitch(
                            CONFIG,
                            xy=players_pitch_xy[team1_mask],
                            face_color=sv.Color.from_hex('#FF1493'),
                            edge_color=sv.Color.from_hex('#ffffff'),
                            radius=8,
                            padding=RADAR_PADDING,
                            scale=RADAR_SCALE,
                            pitch=radar_frame
                        ), radar_frame)

                if len(goalkeepers_detections) > 0:
                    gk_xy = goalkeepers_detections.get_anchors_coordinates(
                        sv.Position.BOTTOM_CENTER)
                    gk_pitch_xy = transformer.transform_points(points=gk_xy)
                    for pt, cid in zip(gk_pitch_xy,
                                       goalkeepers_detections.class_id):
                        col = (sv.Color.from_hex('#00BFFF')
                               if cid == 0
                               else sv.Color.from_hex('#FF1493'))
                        radar_frame = safe_draw(draw_points_on_pitch(
                            CONFIG,
                            xy=pt[np.newaxis],
                            face_color=col,
                            edge_color=sv.Color.from_hex('#000000'),
                            radius=10,
                            padding=RADAR_PADDING,
                            scale=RADAR_SCALE,
                            pitch=radar_frame
                        ), radar_frame)

                if len(referees_detections) > 0:
                    ref_xy = referees_detections.get_anchors_coordinates(
                        sv.Position.BOTTOM_CENTER)
                    ref_pitch_xy = transformer.transform_points(
                        points=ref_xy)
                    radar_frame = safe_draw(draw_points_on_pitch(
                        CONFIG,
                        xy=ref_pitch_xy,
                        face_color=sv.Color.from_hex('#FFD700'),
                        edge_color=sv.Color.from_hex('#000000'),
                        radius=8,
                        padding=RADAR_PADDING,
                        scale=RADAR_SCALE,
                        pitch=radar_frame
                    ), radar_frame)

                if len(ball_detections) > 0:
                    ball_xy = ball_detections.get_anchors_coordinates(
                        sv.Position.CENTER)
                    ball_pitch_xy = transformer.transform_points(
                        points=ball_xy)
                    if ball_pitch_xy.shape[0] == 1:
                        ball_trail.append(ball_pitch_xy.flatten())
                    else:
                        ball_trail.append(
                            np.empty((0,), dtype=np.float32))
                    radar_frame = safe_draw(draw_points_on_pitch(
                        CONFIG,
                        xy=ball_pitch_xy,
                        face_color=sv.Color.from_hex('#ffffff'),
                        edge_color=sv.Color.from_hex('#000000'),
                        radius=6,
                        padding=RADAR_PADDING,
                        scale=RADAR_SCALE,
                        pitch=radar_frame
                    ), radar_frame)
                else:
                    ball_trail.append(np.empty((0,), dtype=np.float32))

                if len(ball_trail) > 1:
                    radar_frame = safe_draw(draw_paths_on_pitch(
                        config=CONFIG,
                        paths=[list(ball_trail)],
                        color=sv.Color.from_hex('#ffffff'),
                        pitch=radar_frame,
                        padding=RADAR_PADDING,
                        scale=RADAR_SCALE
                    ), radar_frame)

            # ── Highlight offside players on radar ────────────────────────────
            if transformer is not None and offside_mask.any():
                offside_xy = players_detections[offside_mask]\
                    .get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                offside_pitch_xy = transformer.transform_points(
                    points=offside_xy)
                radar_frame = safe_draw(draw_points_on_pitch(
                    CONFIG,
                    xy=offside_pitch_xy,
                    face_color=sv.Color.from_hex('#FF0000'),
                    edge_color=sv.Color.from_hex('#ffffff'),
                    radius=10,
                    padding=RADAR_PADDING,
                    scale=RADAR_SCALE,
                    pitch=radar_frame
                ), radar_frame)

            radar_frame = draw_offside_lines_on_radar(
                radar_frame=radar_frame,
                offside_line_x=offside_line_x,
                ball_pitch_x=ball_pitch_x,
                config=CONFIG,
                padding=RADAR_PADDING,
                scale=RADAR_SCALE,
                offside_detected=bool(offside_mask.any()),
            )

            # ── Pad + hstack ──────────────────────────────────────────────────
            annotated_frame = pad_to_height(annotated_frame, out_height)
            radar_frame     = pad_to_height(radar_frame, out_height)

            combined = np.hstack([annotated_frame, radar_frame])
            sink.write_frame(combined)

    verdict = "OFFSIDE" if len(involvement_log) > 0 else "ONSIDE"

    return {
        "verdict":         verdict,
        "involvement_log": involvement_log,
        "output_path":     output_path,
    }
