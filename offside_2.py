import numpy as np
import cv2
import supervision as sv
from sports.configs.soccer import SoccerPitchConfiguration

POSSESSION_THRESHOLD = 0.5

def get_attacking_direction(
    goalkeepers: sv.Detections,
    transformer,
    config: SoccerPitchConfiguration
) -> dict | None:
    """
    Gate: only proceed if a GK is visible.
    Determines which team is attacking and in which direction.
    """
    if transformer is None or len(goalkeepers) == 0:
        return None

    pitch_mid_x = config.length / 2

    gk_xy       = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    gk_pitch_xy = transformer.transform_points(points=gk_xy)

    gk_x          = gk_pitch_xy[:, 0]
    dist_to_left  = np.abs(gk_x - 0)
    dist_to_right = np.abs(gk_x - config.length)
    closest       = np.minimum(dist_to_left, dist_to_right)
    best_idx      = np.argmin(closest)

    gk_pos_x       = gk_pitch_xy[best_idx, 0]
    defending_team = int(goalkeepers.class_id[best_idx])
    attacking_team = 1 - defending_team

    if gk_pos_x < pitch_mid_x:
        defending_goal_x = 0.0
        attack_direction = 'right'
    else:
        defending_goal_x = float(config.length)
        attack_direction = 'left'

    return {
        'defending_team_id': defending_team,
        'attacking_team_id': attacking_team,
        'defending_goal_x':  defending_goal_x,
        'attack_direction':  attack_direction,
    }


def get_offside_line_x(
    players: sv.Detections,
    goalkeepers: sv.Detections,
    transformer,
    direction_info: dict | None,
    config: SoccerPitchConfiguration
) -> float | None:
    """
    Finds x of the 2nd last defender = offside line.
    Needs at least 2 defenders (including GK) visible.
    """
    if direction_info is None or transformer is None:
        return None

    defending_team_id = direction_info['defending_team_id']
    attack_direction  = direction_info['attack_direction']
    defending_goal_x  = direction_info['defending_goal_x']

    all_x = []

    def_mask = players.class_id == defending_team_id
    if def_mask.any():
        def_xy    = players[def_mask].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        def_pitch = transformer.transform_points(points=def_xy)
        all_x.append(def_pitch[:, 0])

    gk_mask = goalkeepers.class_id == defending_team_id
    if gk_mask.any():
        gk_xy    = goalkeepers[gk_mask].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        gk_pitch = transformer.transform_points(points=gk_xy)
        all_x.append(gk_pitch[:, 0])

    if not all_x:
        return None

    all_x_flat = np.concatenate(all_x)

    if len(all_x_flat) < 2:
        return None

    # sort by distance to defending goal — closest first
    dist_to_goal = np.abs(all_x_flat - defending_goal_x)
    sorted_x     = all_x_flat[np.argsort(dist_to_goal)]

    # index 0 = last man (GK), index 1 = 2nd last = offside line
    return float(sorted_x[1])


def check_offside(
    players: sv.Detections,
    transformer,
    direction_info: dict | None,
    offside_line_x: float | None,
    ball_pitch_x: float | None,
    offside_tracker_ids: set,          # persistent set — passed in, mutated, returned
) -> tuple[np.ndarray, set]:
    """
    Distance-based offside check.

    A player is offside if:
      dist(attacker, goal) < dist(offside_line, goal)   [ahead of 2nd last defender]
      AND
      dist(attacker, goal) < dist(ball, goal)            [ahead of the ball]

    Once a tracker_id is added to offside_tracker_ids it stays there forever.

    Returns:
      - offside_mask : bool array over `players`
      - offside_tracker_ids : updated persistent set
    """
    n            = len(players)
    offside_mask = np.zeros(n, dtype=bool)

    if direction_info is None or transformer is None or offside_line_x is None:
        # still mark previously flagged players
        if players.tracker_id is not None:
            for i, tid in enumerate(players.tracker_id):
                if tid in offside_tracker_ids:
                    offside_mask[i] = True
        return offside_mask, offside_tracker_ids

    attacking_team_id = direction_info['attacking_team_id']
    defending_goal_x  = direction_info['defending_goal_x']

    att_mask = players.class_id == attacking_team_id
    if not att_mask.any():
        return offside_mask, offside_tracker_ids

    att_players   = players[att_mask]
    att_xy        = att_players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    att_pitch_xy  = transformer.transform_points(points=att_xy)
    att_x         = att_pitch_xy[:, 0]

    # distances to the defending goal
    dist_attacker_to_goal   = np.abs(att_x         - defending_goal_x)
    dist_offsideline_to_goal = abs(offside_line_x  - defending_goal_x)
    dist_ball_to_goal        = abs(ball_pitch_x    - defending_goal_x) \
                               if ball_pitch_x is not None else float('inf')

    ahead_of_defender = dist_attacker_to_goal < dist_offsideline_to_goal
    ahead_of_ball     = dist_attacker_to_goal < dist_ball_to_goal

    newly_offside = ahead_of_defender & ahead_of_ball

    # update persistent set with tracker ids
    att_indices = np.where(att_mask)[0]
    for i, idx in enumerate(att_indices):
        tid = players.tracker_id[idx] if players.tracker_id is not None else None
        if newly_offside[i] and tid is not None:
            offside_tracker_ids.add(tid)

    # mark all players (any team) who are in the persistent set
    if players.tracker_id is not None:
        for i, tid in enumerate(players.tracker_id):
            if tid in offside_tracker_ids:
                offside_mask[i] = True

    return offside_mask, offside_tracker_ids


def draw_offside_lines_on_radar(
    radar_frame: np.ndarray,
    offside_line_x: float | None,
    ball_pitch_x: float | None,
    config: SoccerPitchConfiguration,
    padding: int,
    scale: float,
    offside_detected: bool = False,
) -> np.ndarray:
    """
    Draws on the radar:
      - Red solid line        → offside line (2nd last defender x)
      - Yellow dashed line    → ball x position
      - Verdict text          → OFFSIDE (red) or ONSIDE (green)
    """
    h, w = radar_frame.shape[:2]

    def pitch_x_to_pixel(px: float) -> int:
        return int(px * scale + padding)

    # yellow dashed — ball
    if ball_pitch_x is not None:
        bx = pitch_x_to_pixel(ball_pitch_x)
        if 0 <= bx < w:
            dash, gap = 10, 10
            for y in range(0, h, dash + gap):
                cv2.line(radar_frame, (bx, y), (bx, min(y + dash, h)),
                         color=(0, 215, 255), thickness=2)

    # red solid — offside line + verdict
    if offside_line_x is not None:
        ox = pitch_x_to_pixel(offside_line_x)
        if 0 <= ox < w:
            cv2.line(radar_frame, (ox, 0), (ox, h),
                     color=(0, 0, 255), thickness=2)

        text  = 'OFFSIDE' if offside_detected else 'ONSIDE'
        color = (0, 0, 255) if offside_detected else (0, 200, 0)
        cv2.putText(
            radar_frame, text,
            org=(padding + 4, h - padding - 4),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA
        )

    return radar_frame

def get_ball_possessor(
    players: sv.Detections,
    goalkeepers: sv.Detections,
    transformer,
    ball_pitch_x: float | None,
    ball_pitch_y: float | None,
) -> int | None:
    """
    Returns the tracker_id of the player closest to the ball,
    or None if no one is within POSSESSION_THRESHOLD.
    """
    if transformer is None or ball_pitch_x is None or ball_pitch_y is None:
        return None

    ball_pos = np.array([ball_pitch_x, ball_pitch_y])
    best_tid  = None
    best_dist = POSSESSION_THRESHOLD  # acts as the cutoff

    for detections in [players, goalkeepers]:
        if len(detections) == 0:
            continue
        xy       = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_xy = transformer.transform_points(points=xy)
        dists    = np.linalg.norm(pitch_xy - ball_pos, axis=1)
        idx      = np.argmin(dists)
        if dists[idx] < best_dist:
            best_dist = dists[idx]
            best_tid  = int(detections.tracker_id[idx])

    return best_tid

def check_offside_involvement(
    current_possessor: int | None,
    previous_possessor: int | None,
    offside_tracker_ids: set,
    involvement_log: list,          # persistent list — passed in, mutated
    frame_number: int,
) -> tuple[bool, list]:
    """
    Detects if ball possession just transferred TO an offside player.
    This covers both:
      - direct receipt (attacker runs onto ball)
      - through pass (teammate passes to offside attacker)
    Both look identical here: previous_possessor != current_possessor
    AND current_possessor is in offside_tracker_ids.

    Returns:
      - involved_this_frame : True if an offside involvement just happened
      - involvement_log     : updated log of all involvement events
    """
    involved_this_frame = False

    if (
        current_possessor is not None
        and current_possessor != previous_possessor
        and current_possessor in offside_tracker_ids
    ):
        involved_this_frame = True
        involvement_log.append({
            'frame':        frame_number,
            'tracker_id':   current_possessor,
            'from_player':  previous_possessor,   # None = ball was loose
        })

    return involved_this_frame, involvement_log