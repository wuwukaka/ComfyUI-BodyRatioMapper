import numpy as np
import math
from itertools import permutations


def select_anchor(batch_pose_data, conf_thresh, has_pt, get_dist, logger=print):
    """
    Select the anchor frame with a two-level WSCS strategy and an optional Z-axis refinement pass.

    Inputs:
    - batch_pose_data: per-frame pose/face dictionaries from the detector.
    - conf_thresh: confidence threshold used for "point exists and valid" checks.
    - has_pt: callback that checks whether a keypoint coordinate is available.
    - get_dist: callback for Euclidean distance between two points.
    - logger: logging function (defaults to print).

    Returns:
    - anchor_idx: selected frame index after WSCS (+ possible Z-axis refinement).
    - best_score: WSCS score of the selected anchor source stage.
    - found_perfect_frame: whether Level-1 had at least one fully valid frame.
    - found_degraded_frame: whether Level-2 had at least one valid degraded frame.
    - level1_scores: list[(frame_idx, score)] for Level-1 valid frames.
    - level2_scores: list[(frame_idx, score)] for Level-2 valid frames.
    """

    # Validate strict face geometry:
    # 1) Nose must be lower than both eyes (Y comparison in image coordinates).
    # 2) Nose X must lie between both eyes.
    # 3) Nose X must lie between both ears.
    # This acts as a hard reject rule for implausible face ordering.
    def is_head_geometry_valid(c):
        if not (has_pt(c[0]) and has_pt(c[14]) and has_pt(c[15]) and has_pt(c[16]) and has_pt(c[17])):
            return False
        x_nose, y_nose = c[0][0], c[0][1]
        x_e1, y_e1 = c[14][0], c[14][1]
        x_e2, y_e2 = c[15][0], c[15][1]
        x_r1 = c[16][0]
        x_r2 = c[17][0]
        cond_y = (y_nose > y_e1) and (y_nose > y_e2)
        cond_x_eyes = min(x_e1, x_e2) <= x_nose <= max(x_e1, x_e2)
        cond_x_ears = min(x_r1, x_r2) <= x_nose <= max(x_r1, x_r2)
        return cond_y and cond_x_eyes and cond_x_ears

    # Reject frames where any right-side limb keypoint crosses to the left of
    # its left-side counterpart on the X axis (or equivalently vice versa).
    def is_limb_folded_by_x_cross(c):
        lr_pairs = [
            (2, 5),   # right/left shoulder
            (3, 6),   # right/left elbow
            (4, 7),   # right/left wrist
            (8, 11),  # right/left hip
            (9, 12),  # right/left knee
            (10, 13), # right/left ankle
        ]
        for r_idx, l_idx in lr_pairs:
            if has_pt(c[r_idx]) and has_pt(c[l_idx]):
                if c[r_idx][0] < c[l_idx][0]:
                    return True
        return False

    # Reject frames when the knee-connection line is too tilted from horizontal.
    def is_knee_tilt_excessive(c, max_deg=30.0):
        if not (has_pt(c[9]) and has_pt(c[12])):
            return False
        dx = abs(c[12][0] - c[9][0])
        dy = abs(c[12][1] - c[9][1])
        if dx <= 1e-6:
            angle_deg = 90.0
        else:
            angle_deg = math.degrees(math.atan2(dy, dx))
        return angle_deg > max_deg

    # Reject frames when the ankle-connection line is too tilted from horizontal.
    def is_ankle_tilt_excessive(c, max_deg=45.0):
        if not (has_pt(c[10]) and has_pt(c[13])):
            return False
        dx = abs(c[13][0] - c[10][0])
        dy = abs(c[13][1] - c[10][1])
        if dx <= 1e-6:
            angle_deg = 90.0
        else:
            angle_deg = math.degrees(math.atan2(dy, dx))
        return angle_deg > max_deg

    # Reject frames when the wrist-connection line is too tilted from horizontal.
    def is_wrist_tilt_excessive(c, max_deg=35.0):
        if not (has_pt(c[4]) and has_pt(c[7])):
            return False
        dx = abs(c[7][0] - c[4][0])
        dy = abs(c[7][1] - c[4][1])
        if dx <= 1e-6:
            angle_deg = 90.0
        else:
            angle_deg = math.degrees(math.atan2(dy, dx))
        return angle_deg > max_deg

    # Reject frames when the elbow-connection line is too tilted from horizontal.
    def is_elbow_tilt_excessive(c, max_deg=35.0):
        if not (has_pt(c[3]) and has_pt(c[6])):
            return False
        dx = abs(c[6][0] - c[3][0])
        dy = abs(c[6][1] - c[3][1])
        if dx <= 1e-6:
            angle_deg = 90.0
        else:
            angle_deg = math.degrees(math.atan2(dy, dx))
        return angle_deg > max_deg

    # Reject frames where wrist lies between shoulder and elbow on Y-axis,
    # indicating possible arm folding/foreshortening.
    def is_wrist_folded(c, shoulder_idx, elbow_idx, wrist_idx):
        if not (has_pt(c[shoulder_idx]) and has_pt(c[elbow_idx]) and has_pt(c[wrist_idx])):
            return False
        sy, ey, wy = c[shoulder_idx][1], c[elbow_idx][1], c[wrist_idx][1]
        return min(sy, ey) < wy < max(sy, ey)

    # Reject frames when ears are above same-side eyes (image Y axis downward).
    def are_ears_above_same_side_eyes(c):
        if not (has_pt(c[14]) and has_pt(c[15]) and has_pt(c[16]) and has_pt(c[17])):
            return False
        left_ear_above_left_eye = c[17][1] < c[15][1]
        right_ear_above_right_eye = c[16][1] < c[14][1]
        return left_ear_above_left_eye and right_ear_above_right_eye

    # Reject when ear-line horizontal angle is excessively larger than eye-line angle.
    # Rule: angle(ears) > ratio_limit * angle(eyes)  => reject
    def is_ear_eye_tilt_ratio_excessive(c, ratio_limit=1.7, ear_angle_min_deg=15.0):
        if not (has_pt(c[14]) and has_pt(c[15]) and has_pt(c[16]) and has_pt(c[17])):
            return False

        def horizontal_small_angle_deg(a, b):
            dx = abs(float(b[0]) - float(a[0]))
            dy = abs(float(b[1]) - float(a[1]))
            if dx <= 1e-6:
                return 90.0
            return math.degrees(math.atan2(dy, dx))

        angle_eyes = horizontal_small_angle_deg(c[14], c[15])
        angle_ears = horizontal_small_angle_deg(c[16], c[17])
        if angle_ears <= ear_angle_min_deg:
            return False
        return angle_ears > (ratio_limit * angle_eyes)

    # Check that every required point both exists and passes the confidence threshold.
    def are_body_points_valid(c, conf, indices):
        for pi in indices:
            if not has_pt(c[pi]) or conf[pi] < conf_thresh:
                return False
        return True

    # Require all 68 face landmarks to pass the threshold (strict face completeness).
    def is_face_conf_valid(face_conf):
        for fi in range(68):
            if fi >= len(face_conf) or face_conf[fi] < conf_thresh:
                return False
        return True

    # Compute mean confidence on a selected subset of body points.
    # This confidence factor scales the final WSCS score.
    def compute_avg_conf(conf, indices):
        valid_conf_sum = 0.0
        valid_conf_count = 0
        for ci in indices:
            if ci < len(conf):
                valid_conf_sum += conf[ci]
                valid_conf_count += 1
        return valid_conf_sum / valid_conf_count if valid_conf_count > 0 else 0.0

    # Positive WSCS geometry score:
    # torso height, shoulder width, and ear width boost the score.
    # Ear width has the highest gain to favor frontal, less-foreshortened heads.
    def compute_base_score_geom(h_torso, w_shoulder, w_ear, h_neck_nose):
        return (h_torso * 1.5) + (w_shoulder * 1.0) + (w_ear * 5.0) + (h_neck_nose * 10.0)

    # Shared penalty score:
    # asymmetry/level/centering penalties for shoulders, arms, legs, and facial alignment.
    # Larger penalty means lower final WSCS score.
    def compute_base_score_penalty(
        p_sh_asym,
        p_elbow_level, p_wrist_level, p_elbow_center, p_wrist_center,
        p_ankle_level, p_knee_level, p_knee_center, p_ankle_center,
        p_eyes_level, p_ears_level, p_ears_align, p_eyes_align, p_nose_align,
        p_ears_nose_align, p_eyes_nose_align,
        p_wrist_angle, p_body_ankle_angle
    ):
        score_penalty = (p_sh_asym * 9.5) + \
                        (p_elbow_level * 12.5) + (p_wrist_level * 14.0) + \
                        (p_elbow_center * 3.0) + (p_wrist_center * 6.5) + \
                        (p_ankle_level * 6.5) + (p_knee_level * 6.0) + \
                        (p_knee_center * 2.5) + (p_ankle_center * 3.5)
        score_penalty += (p_eyes_level * 10.0) + (p_ears_level * 10.0) + \
                         (p_ears_align * 7.5) + (p_eyes_align * 7.5) + (p_nose_align * 14.5) + \
                         (p_ears_nose_align * 14.5) + (p_eyes_nose_align * 14.5)
        score_penalty += (p_wrist_angle * 9.0) + (p_body_ankle_angle * 5.0)
        return score_penalty

    # Keypoint groups for confidence averaging and Level-2 missing penalties.
    # Level-2 assigns larger missing penalties to structurally critical points.
    wscs_conf_indices = [0, 1, 2, 5, 8, 11, 3, 4, 6, 7, 9, 10, 12, 13]
    strict_required_points_l2 = [0, 14, 15, 16, 17, 1, 2, 5, 8, 11]
    wscs_critical_points = [1, 2, 5, 8, 11]
    wscs_head_points = [0, 14, 15, 16, 17]
    wscs_arm_points = [3, 4, 6, 7]
    wscs_leg_points = [9, 10, 12, 13]

    # -------------------------
    # Vectorized WSCS precompute
    # -------------------------
    n_frames = len(batch_pose_data)
    body_xy = np.zeros((n_frames, 18, 2), dtype=float)
    body_conf = np.zeros((n_frames, 18), dtype=float)
    face_conf68 = np.zeros((n_frames, 68), dtype=float)

    for k in range(n_frames):
        frame = batch_pose_data[k]
        c = frame['bodies']['candidate']
        conf = frame['bodies']['candidate_conf']
        face_conf = frame['faces_conf'][0] if len(frame['faces_conf']) > 0 else np.zeros((68,))

        c_rows = min(18, len(c))
        conf_rows = min(18, len(conf))
        face_rows = min(68, len(face_conf))

        if c_rows > 0:
            body_xy[k, :c_rows] = np.asarray(c[:c_rows], dtype=float)
        if conf_rows > 0:
            body_conf[k, :conf_rows] = np.asarray(conf[:conf_rows], dtype=float)
        if face_rows > 0:
            face_conf68[k, :face_rows] = np.asarray(face_conf[:face_rows], dtype=float)

    x = body_xy[:, :, 0]
    y = body_xy[:, :, 1]
    pt_present = np.sum(np.abs(body_xy), axis=2) > 0.01
    conf_ok = body_conf >= conf_thresh
    pt_valid = pt_present & conf_ok
    face_conf_ok = np.all(face_conf68 >= conf_thresh, axis=1)

    def pair_dist(i, j):
        return np.sqrt((x[:, i] - x[:, j]) ** 2 + (y[:, i] - y[:, j]) ** 2)

    def pair_angle_and_mask(i, j):
        mask = pt_present[:, i] & pt_present[:, j]
        dx = np.abs(x[:, j] - x[:, i])
        dy = np.abs(y[:, j] - y[:, i])
        angle = np.degrees(np.arctan2(dy, np.maximum(dx, 1e-6)))
        return angle, mask

    head_has = pt_present[:, 0] & pt_present[:, 14] & pt_present[:, 15] & pt_present[:, 16] & pt_present[:, 17]
    cond_y = (y[:, 0] > y[:, 14]) & (y[:, 0] > y[:, 15])
    cond_x_eyes = (x[:, 0] >= np.minimum(x[:, 14], x[:, 15])) & (x[:, 0] <= np.maximum(x[:, 14], x[:, 15]))
    cond_x_ears = (x[:, 0] >= np.minimum(x[:, 16], x[:, 17])) & (x[:, 0] <= np.maximum(x[:, 16], x[:, 17]))
    head_geometry_valid = head_has & cond_y & cond_x_eyes & cond_x_ears

    ankle_angle, ankle_pair = pair_angle_and_mask(10, 13)
    wrist_angle_line, wrist_pair = pair_angle_and_mask(4, 7)
    elbow_angle_line, elbow_pair = pair_angle_and_mask(3, 6)
    is_ankle_tilt_excessive_arr = ankle_pair & (ankle_angle > 45.0)
    is_wrist_tilt_excessive_arr = wrist_pair & (wrist_angle_line > 35.0)
    is_elbow_tilt_excessive_arr = elbow_pair & (elbow_angle_line > 35.0)

    # wrist between shoulder and elbow on Y-axis => folding cue
    r_fold_has = pt_present[:, 2] & pt_present[:, 3] & pt_present[:, 4]
    l_fold_has = pt_present[:, 5] & pt_present[:, 6] & pt_present[:, 7]
    r_fold = r_fold_has & (np.minimum(y[:, 2], y[:, 3]) < y[:, 4]) & (y[:, 4] < np.maximum(y[:, 2], y[:, 3]))
    l_fold = l_fold_has & (np.minimum(y[:, 5], y[:, 6]) < y[:, 7]) & (y[:, 7] < np.maximum(y[:, 5], y[:, 6]))
    is_wrist_folded_arr = r_fold | l_fold

    ears_eyes_has = pt_present[:, 14] & pt_present[:, 15] & pt_present[:, 16] & pt_present[:, 17]
    are_ears_above_same_side_eyes_arr = ears_eyes_has & (y[:, 17] < y[:, 15]) & (y[:, 16] < y[:, 14])

    eyes_angle, eyes_pair = pair_angle_and_mask(14, 15)
    ears_angle, ears_pair = pair_angle_and_mask(16, 17)
    is_ear_eye_tilt_ratio_excessive_arr = eyes_pair & ears_pair & (ears_angle > 15.0) & (ears_angle > (1.7 * eyes_angle))

    common_hard_reject = (
        is_ankle_tilt_excessive_arr
        | is_wrist_tilt_excessive_arr
        | is_elbow_tilt_excessive_arr
        | is_wrist_folded_arr
        | are_ears_above_same_side_eyes_arr
        | is_ear_eye_tilt_ratio_excessive_arr
    )

    has_shoulders = pt_present[:, 2] & pt_present[:, 5]
    has_hips = pt_present[:, 8] & pt_present[:, 11]
    mid_sh_x = (x[:, 2] + x[:, 5]) * 0.5
    mid_sh_y = (y[:, 2] + y[:, 5]) * 0.5
    mid_hip_x = (x[:, 8] + x[:, 11]) * 0.5
    mid_hip_y = (y[:, 8] + y[:, 11]) * 0.5
    h_torso = np.where(has_shoulders & has_hips, np.sqrt((mid_sh_x - mid_hip_x) ** 2 + (mid_sh_y - mid_hip_y) ** 2), 0.0)

    w_shoulder = pair_dist(2, 5)
    h_neck_nose = np.where(pt_present[:, 1] & pt_present[:, 0], np.abs(y[:, 1] - y[:, 0]), 0.0)
    dist_neck_r = np.where(pt_present[:, 1] & pt_present[:, 2], pair_dist(1, 2), 0.0)
    dist_neck_l = np.where(pt_present[:, 1] & pt_present[:, 5], pair_dist(1, 5), 0.0)
    p_sh_asym = np.abs(dist_neck_r - dist_neck_l)

    p_elbow_level = np.full((n_frames,), 0.5, dtype=float)
    p_elbow_center = np.full((n_frames,), 0.5, dtype=float)
    elbow_has = pt_present[:, 3] & pt_present[:, 6]
    p_elbow_level[elbow_has] = np.abs(y[elbow_has, 3] - y[elbow_has, 6])
    p_elbow_center[elbow_has] = np.abs((x[elbow_has, 3] + x[elbow_has, 6]) * 0.5 - x[elbow_has, 1])

    p_wrist_level = np.full((n_frames,), 0.5, dtype=float)
    p_wrist_center = np.full((n_frames,), 0.5, dtype=float)
    wrist_has = pt_present[:, 4] & pt_present[:, 7]
    p_wrist_level[wrist_has] = np.abs(y[wrist_has, 4] - y[wrist_has, 7])
    p_wrist_center[wrist_has] = np.abs((x[wrist_has, 4] + x[wrist_has, 7]) * 0.5 - x[wrist_has, 1])

    p_knee_level = np.full((n_frames,), 0.5, dtype=float)
    p_knee_center = np.full((n_frames,), 0.5, dtype=float)
    knee_has = pt_present[:, 9] & pt_present[:, 12]
    p_knee_level[knee_has] = np.abs(y[knee_has, 9] - y[knee_has, 12])
    knee_center_has = knee_has & has_hips
    p_knee_center[knee_center_has] = np.abs((x[knee_center_has, 9] + x[knee_center_has, 12]) * 0.5 - mid_hip_x[knee_center_has])

    p_ankle_level = np.full((n_frames,), 0.5, dtype=float)
    p_ankle_center = np.full((n_frames,), 0.5, dtype=float)
    ankle_has = pt_present[:, 10] & pt_present[:, 13]
    p_ankle_level[ankle_has] = np.abs(y[ankle_has, 10] - y[ankle_has, 13])
    ankle_center_has = ankle_has & has_hips
    p_ankle_center[ankle_center_has] = np.abs((x[ankle_center_has, 10] + x[ankle_center_has, 13]) * 0.5 - mid_hip_x[ankle_center_has])

    c_conf = np.mean(body_conf[:, wscs_conf_indices], axis=1)
    p_eyes_level = np.where(pt_present[:, 14] & pt_present[:, 15], np.abs(y[:, 14] - y[:, 15]), 0.0)
    p_ears_level = np.where(pt_present[:, 16] & pt_present[:, 17], np.abs(y[:, 16] - y[:, 17]), 0.0)
    p_eyes_align = np.where(pt_present[:, 1] & pt_present[:, 14] & pt_present[:, 15], np.abs((x[:, 14] + x[:, 15]) * 0.5 - x[:, 1]), 0.0)
    p_ears_align = np.where(pt_present[:, 1] & pt_present[:, 16] & pt_present[:, 17], np.abs((x[:, 16] + x[:, 17]) * 0.5 - x[:, 1]), 0.0)
    p_nose_align = np.where(pt_present[:, 1] & pt_present[:, 0], np.abs(x[:, 0] - x[:, 1]), 0.0)
    p_ears_nose_align = np.where(pt_present[:, 0] & pt_present[:, 16] & pt_present[:, 17], np.abs((x[:, 16] + x[:, 17]) * 0.5 - x[:, 0]), 0.0)
    p_eyes_nose_align = np.where(pt_present[:, 0] & pt_present[:, 14] & pt_present[:, 15], np.abs((x[:, 14] + x[:, 15]) * 0.5 - x[:, 0]), 0.0)
    p_wrist_angle = np.where(wrist_pair, wrist_angle_line, 0.0)
    p_body_ankle_angle = np.where(ankle_pair, ankle_angle, 0.0)

    w_ear = np.where(pt_present[:, 16] & pt_present[:, 17], pair_dist(16, 17), 0.0)
    score_geom_arr = (h_torso * 1.5) + (w_shoulder * 1.0) + (w_ear * 5.0) + (h_neck_nose * 10.0)
    score_penalty_arr = (
        (p_sh_asym * 9.5) +
        (p_elbow_level * 12.5) + (p_wrist_level * 14.0) +
        (p_elbow_center * 3.0) + (p_wrist_center * 6.5) +
        (p_ankle_level * 6.5) + (p_knee_level * 6.0) +
        (p_knee_center * 2.5) + (p_ankle_center * 3.5) +
        (p_eyes_level * 10.0) + (p_ears_level * 10.0) +
        (p_ears_align * 7.5) + (p_eyes_align * 7.5) + (p_nose_align * 14.5) +
        (p_ears_nose_align * 14.5) + (p_eyes_nose_align * 14.5) +
        (p_wrist_angle * 9.0) + (p_body_ankle_angle * 5.0)
    )
    score_base_arr = (score_geom_arr - score_penalty_arr) * c_conf

    missing_weights = np.full((18,), 3.5, dtype=float)
    missing_weights[wscs_critical_points] = 6.5
    missing_weights[wscs_head_points] = 5.5
    missing_weights[wscs_arm_points] = 4.5
    missing_weights[wscs_leg_points] = 4.0
    missing_mask = (~pt_present) | (body_conf < conf_thresh)
    missing_penalty_arr = np.sum(missing_mask * missing_weights[np.newaxis, :], axis=1)

    level1_valid_mask = (
        np.all(pt_valid[:, :18], axis=1)
        & face_conf_ok
        & head_geometry_valid
        & (~common_hard_reject)
    )
    level2_valid_mask = (
        np.all(pt_valid[:, strict_required_points_l2], axis=1)
        & face_conf_ok
        & head_geometry_valid
        & (~common_hard_reject)
    )

    # Level-1 (strict):
    # - Requires all 18 body points valid + all 68 face points valid + head geometry valid.
    # - Scores each valid frame with:
    #   (positive geometry - weighted penalties) * average confidence.
    # - Picks the max-score frame.
    def run_level1_wscs_scoring():
        local_best_score = -float('inf')
        local_best_anchor_idx = 0
        local_found_perfect_frame = False
        local_level1_scores = []
        valid_indices = np.flatnonzero(level1_valid_mask)
        if len(valid_indices) > 0:
            local_found_perfect_frame = True
            for k in valid_indices:
                score = float(score_base_arr[k])
                local_level1_scores.append((int(k), score))
                if score > local_best_score:
                    local_best_score = score
                    local_best_anchor_idx = int(k)

        return local_found_perfect_frame, local_level1_scores, local_best_score, local_best_anchor_idx

    # Level-2 fallback (degraded):
    # - Requires a strict subset of body points + full 68 face confidence + head geometry.
    # - Uses the same base score, then subtracts explicit per-point missing penalties.
    # - Allows anchor search to continue when Level-1 finds no fully valid frame.
    def run_level2_wscs_scoring():
        local_best_score_l2 = -float('inf')
        local_best_anchor_idx_l2 = 0
        local_found_degraded_frame = False
        local_level2_scores = []
        valid_indices = np.flatnonzero(level2_valid_mask)
        if len(valid_indices) > 0:
            local_found_degraded_frame = True
            l2_score_arr = (score_geom_arr - score_penalty_arr - missing_penalty_arr) * c_conf
            for k in valid_indices:
                score = float(l2_score_arr[k])
                local_level2_scores.append((int(k), score))
                if score > local_best_score_l2:
                    local_best_score_l2 = score
                    local_best_anchor_idx_l2 = int(k)

        return local_found_degraded_frame, local_level2_scores, local_best_score_l2, local_best_anchor_idx_l2

    # Two-stage controller:
    # 1) Try Level-1 strict search.
    # 2) If no frame passes Level-1, run Level-2 degraded search.
    def run_two_level_wscs_selection():
        local_best_score = -float('inf')
        local_best_anchor_idx = 0
        local_found_perfect_frame = False
        local_found_degraded_frame = False
        local_level1_scores = []
        local_level2_scores = []

        local_found_perfect_frame, local_level1_scores, local_best_score, local_best_anchor_idx = run_level1_wscs_scoring()
        if not local_found_perfect_frame:
            logger(f"[WSCS] No perfect frame found in {len(batch_pose_data)} frames. Activating Level-2 fallback (Pure Penalty-Based)...")
            local_found_degraded_frame, local_level2_scores, best_score_l2, best_anchor_idx_l2 = run_level2_wscs_scoring()
            if local_found_degraded_frame:
                logger(f"[WSCS] Level-2 fallback successful: Found idx={best_anchor_idx_l2}, score={best_score_l2:.4f}")
                local_best_score = best_score_l2
                local_best_anchor_idx = best_anchor_idx_l2
            else:
                logger("[WSCS] Level-2 fallback also failed. Using first frame (idx=0) as anchor with caution.")

        return local_best_anchor_idx, local_best_score, local_found_perfect_frame, local_found_degraded_frame, local_level1_scores, local_level2_scores

    # Thin wrapper for selection + status logging.
    def run_auto_anchor_search():
        local_anchor_idx, local_best_score, local_found_perfect_frame, local_found_degraded_frame, local_level1_scores, local_level2_scores = run_two_level_wscs_selection()
        if local_found_perfect_frame:
            logger(f"[WSCS] Level-1 Perfect Frame selected: idx={local_anchor_idx}, score={local_best_score:.4f}")
        elif local_found_degraded_frame:
            logger(f"[WSCS] Level-2 Degraded Frame selected: idx={local_anchor_idx}, score={local_best_score:.4f} (Penalty-Based, Head Lock: PRESERVED)")
        else:
            logger("[WSCS] WARNING: No valid anchor found. Using frame 0 as fallback.")
        return local_anchor_idx, local_best_score, local_found_perfect_frame, local_found_degraded_frame, local_level1_scores, local_level2_scores

    # Z-axis filter consumes Level-1 scores if available; otherwise Level-2 scores.
    def select_wscs_scores_for_z_filter(found_perfect, scores_l1, found_degraded, scores_l2):
        if found_perfect:
            return scores_l1
        if found_degraded:
            return scores_l2
        return []

    # Z-axis foreshortening refinement:
    # - Start from top WSCS candidates.
    # - Remove obvious wrist-folded foreshortened poses.
    # - Rank candidates by multiple normalized body ratios.
    # - Prefer candidates with consistent low foreshortening across torso/limbs.
    def run_z_axis_filter(valid_scores, current_anchor_idx):
        if len(valid_scores) <= 1:
            return current_anchor_idx

        logger(f"[Z-Axis Filter] Starting Z-axis foreshortening filter, initial candidate frames: {len(valid_scores)}")

        def get_rem_count(total, ratio):
            val = total * ratio
            int_val = int(val)
            if val > int_val:
                return int_val + 1
            return max(1, int_val)

        # Stage 0: keep top 15% WSCS frames as geometric candidates.
        valid_scores.sort(key=lambda x: x[1], reverse=True)
        rem1 = get_rem_count(len(valid_scores), 0.15)
        candidates = valid_scores[:rem1]
        c_indices = [x[0] for x in candidates]

        if len(c_indices) >= 2:
            def get_len_safe(c, p1, p2):
                if has_pt(c[p1]) and has_pt(c[p2]):
                    return get_dist(c[p1], c[p2])
                return 0.001

            # Pre-filter: wrist between shoulder and elbow in Y is treated as arm folding cue.
            filtered_indices = []
            for k_idx in c_indices:
                c_k = batch_pose_data[k_idx]['bodies']['candidate']
                filtered_indices.append(k_idx)
            if len(filtered_indices) > 0:
                c_indices = filtered_indices

            # Build normalized inverse-length ratios with a head-width proxy (hw).
            # Smaller ratio generally indicates less depth compression (less fold/foreshortening).
            ratios_dict = {
                'torso': {}, 'shoulder': {},
                'arm_up_l': {}, 'arm_up_r': {},
                'arm_low_l': {}, 'arm_low_r': {},
                'leg_up_l': {}, 'leg_up_r': {},
                'leg_low_l': {}, 'leg_low_r': {}
            }

            for k_idx in c_indices:
                c_k = batch_pose_data[k_idx]['bodies']['candidate']
                if has_pt(c_k[16]) and has_pt(c_k[17]):
                    hw = get_dist(c_k[16], c_k[17])
                elif has_pt(c_k[14]) and has_pt(c_k[15]):
                    hw = get_dist(c_k[14], c_k[15]) * 1.5
                elif has_pt(c_k[0]) and has_pt(c_k[1]):
                    hw = get_dist(c_k[0], c_k[1]) * 0.8
                else:
                    hw = 1.0
                hw = max(hw, 0.001)

                ms = (c_k[2] + c_k[5]) * 0.5 if has_pt(c_k[2]) and has_pt(c_k[5]) else c_k[1]
                mh = (c_k[8] + c_k[11]) * 0.5 if has_pt(c_k[8]) and has_pt(c_k[11]) else c_k[1]
                t_len = get_dist(ms, mh) if has_pt(ms) and has_pt(mh) else 0.001
                ratios_dict['torso'][k_idx] = hw / max(t_len, 0.001)
                ratios_dict['shoulder'][k_idx] = hw / max(get_len_safe(c_k, 2, 5), 0.001)
                ratios_dict['arm_up_l'][k_idx] = hw / max(get_len_safe(c_k, 5, 6), 0.001)
                ratios_dict['arm_up_r'][k_idx] = hw / max(get_len_safe(c_k, 2, 3), 0.001)
                ratios_dict['arm_low_l'][k_idx] = hw / max(get_len_safe(c_k, 6, 7), 0.001)
                ratios_dict['arm_low_r'][k_idx] = hw / max(get_len_safe(c_k, 3, 4), 0.001)
                ratios_dict['leg_up_l'][k_idx] = hw / max(get_len_safe(c_k, 11, 12), 0.001)
                ratios_dict['leg_up_r'][k_idx] = hw / max(get_len_safe(c_k, 8, 9), 0.001)
                ratios_dict['leg_low_l'][k_idx] = hw / max(get_len_safe(c_k, 12, 13), 0.001)
                ratios_dict['leg_low_r'][k_idx] = hw / max(get_len_safe(c_k, 9, 10), 0.001)

            rank_torso = sorted(c_indices, key=lambda x: ratios_dict['torso'][x])
            rank_shoulder = sorted(c_indices, key=lambda x: ratios_dict['shoulder'][x])
            rank_aul = sorted(c_indices, key=lambda x: ratios_dict['arm_up_l'][x])
            rank_aur = sorted(c_indices, key=lambda x: ratios_dict['arm_up_r'][x])
            rank_all = sorted(c_indices, key=lambda x: ratios_dict['arm_low_l'][x])
            rank_alr = sorted(c_indices, key=lambda x: ratios_dict['arm_low_r'][x])
            rank_lul = sorted(c_indices, key=lambda x: ratios_dict['leg_up_l'][x])
            rank_lur = sorted(c_indices, key=lambda x: ratios_dict['leg_up_r'][x])
            rank_lll = sorted(c_indices, key=lambda x: ratios_dict['leg_low_l'][x])
            rank_llr = sorted(c_indices, key=lambda x: ratios_dict['leg_low_r'][x])

            def apply_round2_mixed_filter(indices, ratios):
                if len(indices) < 2:
                    return indices
                rem2 = get_rem_count(len(indices), 0.20)
                rank_t = sorted(indices, key=lambda x: ratios['torso'][x])
                rank_s = sorted(indices, key=lambda x: ratios['shoulder'][x])
                result_intersect = set(rank_t[:rem2]) & set(rank_s[:rem2])
                step2_ts = sorted(rank_t[:rem2], key=lambda x: ratios['shoulder'][x])[:rem2]
                step2_st = sorted(rank_s[:rem2], key=lambda x: ratios['torso'][x])[:rem2]
                return list(result_intersect | set(step2_ts) | set(step2_st))

            def apply_round3_mixed_filter(indices, ratios):
                if len(indices) < 2:
                    return indices
                rem3 = get_rem_count(len(indices), 0.20)
                rank_aul_sub = sorted(indices, key=lambda x: ratios['arm_up_l'][x])
                rank_aur_sub = sorted(indices, key=lambda x: ratios['arm_up_r'][x])
                rank_all_sub = sorted(indices, key=lambda x: ratios['arm_low_l'][x])
                rank_alr_sub = sorted(indices, key=lambda x: ratios['arm_low_r'][x])
                rank_lul_sub = sorted(indices, key=lambda x: ratios['leg_up_l'][x])
                rank_lur_sub = sorted(indices, key=lambda x: ratios['leg_up_r'][x])
                rank_lll_sub = sorted(indices, key=lambda x: ratios['leg_low_l'][x])
                rank_llr_sub = sorted(indices, key=lambda x: ratios['leg_low_r'][x])
                result_intersect = (set(rank_aul_sub[:rem3]) & set(rank_aur_sub[:rem3]) &
                                    set(rank_all_sub[:rem3]) & set(rank_alr_sub[:rem3]) &
                                    set(rank_lul_sub[:rem3]) & set(rank_lur_sub[:rem3]) &
                                    set(rank_lll_sub[:rem3]) & set(rank_llr_sub[:rem3]))
                parts = ['arm_up_l', 'arm_up_r', 'arm_low_l', 'arm_low_r', 'leg_up_l', 'leg_up_r', 'leg_low_l', 'leg_low_r']
                all_results = [result_intersect]
                for perm in permutations(parts):
                    current_set = set(indices)
                    for part in perm:
                        current_set = set(sorted(current_set, key=lambda x: ratios[part][x])[:rem3])
                        if len(current_set) == 0:
                            break
                    all_results.append(current_set)
                return list(set.union(*all_results) if any(all_results) else set())

            # Z-filter Level-1:
            # For gradually increasing top-k percentages, intersect rank fronts of all ratio groups.
            # If non-empty intersection exists, choose the minimum rank-distance candidate.
            found_intersection = False
            final_candidates = []
            for pct in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
                rem = int(len(c_indices) * pct)
                if rem < 1:
                    continue
                intersection = set(rank_torso[:rem]) & set(rank_shoulder[:rem]) & \
                               set(rank_aul[:rem]) & set(rank_aur[:rem]) & \
                               set(rank_all[:rem]) & set(rank_alr[:rem]) & \
                               set(rank_lul[:rem]) & set(rank_lur[:rem]) & \
                               set(rank_lll[:rem]) & set(rank_llr[:rem])
                if len(intersection) > 0:
                    final_candidates = list(intersection)
                    found_intersection = True
                    logger(f"[Z-Axis Filter] Level 1 intersection screening successful, ratio={pct*100:.0f}%, remaining {len(final_candidates)} frames")
                    break

            if found_intersection:
                if len(final_candidates) > 1:
                    best_final_idx = min(final_candidates, key=lambda x:
                        rank_torso.index(x)**2 + rank_shoulder.index(x)**2 +
                        rank_aul.index(x)**2 + rank_aur.index(x)**2 +
                        rank_all.index(x)**2 + rank_alr.index(x)**2 +
                        rank_lul.index(x)**2 + rank_lur.index(x)**2 +
                        rank_lll.index(x)**2 + rank_llr.index(x)**2
                    )
                else:
                    best_final_idx = final_candidates[0]
                current_anchor_idx = best_final_idx
                logger(f"[Z-Axis Filter] Level 1 screening complete, final selection idx={current_anchor_idx}")
            # Z-filter Level-2 mixed strategy:
            # - Apply round2/round3 in two different orders.
            # - Merge survivors from both orders.
            # - Final tie-break includes WSCS rank as an extra weighted term.
            else:
                logger("[Z-Axis Filter] Level 1 intersection is empty, downgrading to mixed strategy")
                c_indices_order1 = apply_round2_mixed_filter(c_indices.copy(), ratios_dict)
                if len(c_indices_order1) >= 2:
                    c_indices_order1 = apply_round3_mixed_filter(c_indices_order1, ratios_dict)

                c_indices_order2 = c_indices.copy()
                if len(c_indices_order2) >= 2:
                    c_indices_order2 = apply_round3_mixed_filter(c_indices_order2, ratios_dict)
                c_indices_order2 = apply_round2_mixed_filter(c_indices_order2, ratios_dict)

                c_indices = list(set(c_indices_order1) | set(c_indices_order2))
                logger(f"[Z-Axis Filter] Mixed strategy: Order1={len(c_indices_order1)} frames, Order2={len(c_indices_order2)} frames, Final={len(c_indices)} frames")

                if len(c_indices) > 0:
                    wscs_scores = {k_idx: score for k_idx, score in candidates if k_idx in c_indices}
                    rank_wscs = sorted(c_indices, key=lambda x: wscs_scores.get(x, 0), reverse=True)
                    best_final_idx = min(c_indices, key=lambda x:
                        rank_torso.index(x)**2 + rank_shoulder.index(x)**2 +
                        rank_aul.index(x)**2 + rank_aur.index(x)**2 +
                        rank_all.index(x)**2 + rank_alr.index(x)**2 +
                        rank_lul.index(x)**2 + rank_lur.index(x)**2 +
                        rank_lll.index(x)**2 + rank_llr.index(x)**2 +
                        rank_wscs.index(x) * 2
                    )
                    current_anchor_idx = best_final_idx
                    logger(f"[Z-Axis Filter] Level 2 screening complete, final selection idx={current_anchor_idx}")
        return current_anchor_idx

    # Main execution path:
    # WSCS picks the candidate set first, then Z-axis filter optionally refines the anchor index.
    anchor_idx, best_score, found_perfect_frame, found_degraded_frame, level1_scores, level2_scores = run_auto_anchor_search()
    valid_scores_list = select_wscs_scores_for_z_filter(found_perfect_frame, level1_scores, found_degraded_frame, level2_scores)
    anchor_idx = run_z_axis_filter(valid_scores_list, anchor_idx)
    return anchor_idx, best_score, found_perfect_frame, found_degraded_frame, level1_scores, level2_scores
