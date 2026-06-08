# Copyright (C) 2026 wuwukasi (wuwukaka)
# SPDX-License-Identifier: GPL-3.0-only

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
    knee_angle_line, knee_pair = pair_angle_and_mask(9, 12)
    shoulder_angle_line, shoulder_pair = pair_angle_and_mask(2, 5)
    is_ankle_tilt_excessive_arr = ankle_pair & (ankle_angle > 45.0)
    is_wrist_tilt_excessive_arr = wrist_pair & (wrist_angle_line > 35.0)
    is_elbow_tilt_excessive_arr = elbow_pair & (elbow_angle_line > 35.0)
    is_knee_tilt_excessive_arr = knee_pair & (knee_angle_line > 15.0)
    is_shoulder_tilt_excessive_arr = shoulder_pair & (shoulder_angle_line > 15.0)

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

    # Reject when ear-to-nose distances are too asymmetric (ratio > 1.2).
    ear_nose_has = pt_present[:, 0] & pt_present[:, 16] & pt_present[:, 17]
    dist_ear_r_nose = np.sqrt((x[:, 16] - x[:, 0]) ** 2 + (y[:, 16] - y[:, 0]) ** 2)
    dist_ear_l_nose = np.sqrt((x[:, 17] - x[:, 0]) ** 2 + (y[:, 17] - y[:, 0]) ** 2)
    max_ear_nose = np.maximum(dist_ear_r_nose, dist_ear_l_nose)
    min_ear_nose = np.minimum(dist_ear_r_nose, dist_ear_l_nose)
    is_ear_nose_ratio_excessive_arr = ear_nose_has & (min_ear_nose > 1e-6) & ((max_ear_nose / min_ear_nose) > 1.26)

    # Reject when nose is above the ear line (interpolated Y at nose X) and nose-to-ear angle deviates from ear line by >10°.
    ear_raw_dx = x[:, 17] - x[:, 16]
    ear_line_y_at_nose = np.where(np.abs(ear_raw_dx) > 1e-6, y[:, 16] + (y[:, 17] - y[:, 16]) * (x[:, 0] - x[:, 16]) / ear_raw_dx, np.minimum(y[:, 16], y[:, 17]))
    nose_above_ears = pt_present[:, 0] & pt_present[:, 16] & pt_present[:, 17] & (y[:, 0] < ear_line_y_at_nose)
    ear_line_dx = np.abs(x[:, 17] - x[:, 16])
    ear_line_dy = np.abs(y[:, 17] - y[:, 16])
    ear_line_angle = np.degrees(np.arctan2(ear_line_dy, np.maximum(ear_line_dx, 1e-6)))
    nose_ear_r_dx = np.abs(x[:, 16] - x[:, 0])
    nose_ear_r_dy = np.abs(y[:, 16] - y[:, 0])
    nose_ear_r_angle = np.degrees(np.arctan2(nose_ear_r_dy, np.maximum(nose_ear_r_dx, 1e-6)))
    nose_ear_l_dx = np.abs(x[:, 17] - x[:, 0])
    nose_ear_l_dy = np.abs(y[:, 17] - y[:, 0])
    nose_ear_l_angle = np.degrees(np.arctan2(nose_ear_l_dy, np.maximum(nose_ear_l_dx, 1e-6)))
    is_nose_above_ears_tilted_arr = nose_above_ears & ((np.abs(nose_ear_r_angle - ear_line_angle) > 10.0) | (np.abs(nose_ear_l_angle - ear_line_angle) > 10.0))
    is_ear_line_tilt_excessive_arr = ears_pair & (ear_line_angle > 25.0)

    # Reject when left/right body segment ratio exceeds 1.24 (torso, upper leg, lower leg).
    def lr_ratio_mask(i1, i2, j1, j2):
        has = pt_present[:, i1] & pt_present[:, i2] & pt_present[:, j1] & pt_present[:, j2]
        d_left = pair_dist(i1, i2)
        d_right = pair_dist(j1, j2)
        max_d = np.maximum(d_left, d_right)
        min_d = np.minimum(d_left, d_right)
        return has & (min_d > 1e-6) & ((max_d / min_d) > 1.24)

    is_torso_ratio_excessive_arr = lr_ratio_mask(1, 11, 1, 8)
    is_upper_leg_ratio_excessive_arr = lr_ratio_mask(11, 12, 8, 9)
    is_lower_leg_ratio_excessive_arr = lr_ratio_mask(12, 13, 9, 10)
    is_body_ratio_excessive_arr = is_torso_ratio_excessive_arr | is_upper_leg_ratio_excessive_arr | is_lower_leg_ratio_excessive_arr

    # Reject when nose is below the shoulder line (interpolated Y at nose X).
    sh_raw_dx = x[:, 5] - x[:, 2]
    sh_line_y_at_nose = np.where(np.abs(sh_raw_dx) > 1e-6, y[:, 2] + (y[:, 5] - y[:, 2]) * (x[:, 0] - x[:, 2]) / sh_raw_dx, np.maximum(y[:, 2], y[:, 5]))
    is_nose_below_shoulder_line_arr = pt_present[:, 0] & pt_present[:, 2] & pt_present[:, 5] & (y[:, 0] >= sh_line_y_at_nose)

    # Reject obvious back-facing poses. For a front-facing person, right-side keypoints
    # appear on the image-left: right_shoulder(2).x < left_shoulder(5).x and right_hip(8).x < left_hip(11).x.
    shoulder_lr_margin = np.maximum(2.0, np.abs(x[:, 5] - x[:, 2]) * 0.05)
    hip_lr_margin = np.maximum(2.0, np.abs(x[:, 11] - x[:, 8]) * 0.05)
    is_back_facing_by_shoulder_arr = (pt_present[:, 2] & pt_present[:, 5] & ((x[:, 5] + shoulder_lr_margin) < x[:, 2]))
    is_back_facing_by_hip_arr = (pt_present[:, 8] & pt_present[:, 11] & ((x[:, 11] + hip_lr_margin) < x[:, 8]))
    is_back_facing_arr = is_back_facing_by_shoulder_arr | is_back_facing_by_hip_arr

    common_hard_reject = (
        is_ankle_tilt_excessive_arr
        | is_wrist_tilt_excessive_arr
        | is_elbow_tilt_excessive_arr
        | is_wrist_folded_arr
        | are_ears_above_same_side_eyes_arr
        | is_ear_eye_tilt_ratio_excessive_arr
        | is_ear_nose_ratio_excessive_arr
        | is_nose_above_ears_tilted_arr
        | is_ear_line_tilt_excessive_arr
        | is_knee_tilt_excessive_arr
        | is_shoulder_tilt_excessive_arr
        | is_body_ratio_excessive_arr
        | is_nose_below_shoulder_line_arr
    )

    has_shoulders = pt_present[:, 2] & pt_present[:, 5]
    has_hips = pt_present[:, 8] & pt_present[:, 11]
    mid_sh_x = (x[:, 2] + x[:, 5]) * 0.5
    mid_sh_y = (y[:, 2] + y[:, 5]) * 0.5
    mid_hip_x = (x[:, 8] + x[:, 11]) * 0.5
    mid_hip_y = (y[:, 8] + y[:, 11]) * 0.5
    h_torso = np.where(has_shoulders & has_hips, np.sqrt((mid_sh_x - mid_hip_x) ** 2 + (mid_sh_y - mid_hip_y) ** 2), 0.0)

    w_shoulder = np.where(pt_present[:, 2] & pt_present[:, 5], np.abs(x[:, 2] - x[:, 5]), 0.0)
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

    w_ear = np.where(pt_present[:, 16] & pt_present[:, 17], np.abs(x[:, 16] - x[:, 17]), 0.0)
    score_geom_arr = (h_torso * 1.5) + (w_shoulder * 3.0) + (w_ear * 8.0) + (h_neck_nose * 10.0)
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

    level1_base_valid_mask = (
        np.all(pt_valid[:, :18], axis=1)
        & face_conf_ok
        & head_geometry_valid
        & (~common_hard_reject)
    )
    level2_base_valid_mask = (
        np.all(pt_valid[:, strict_required_points_l2], axis=1)
        & face_conf_ok
        & head_geometry_valid
        & (~common_hard_reject)
    )
    level1_valid_mask = level1_base_valid_mask & (~is_back_facing_arr)
    if not np.any(level1_valid_mask) and np.any(level1_base_valid_mask):
        logger("[WSCS] Back-facing hard filter removed all Level-1 candidates; falling back to unfiltered Level-1 candidates")
        level1_valid_mask = level1_base_valid_mask

    level2_valid_mask = level2_base_valid_mask & (~is_back_facing_arr)
    if not np.any(level2_valid_mask) and np.any(level2_base_valid_mask):
        logger("[WSCS] Back-facing hard filter removed all Level-2 candidates; falling back to unfiltered Level-2 candidates")
        level2_valid_mask = level2_base_valid_mask

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
                result = set(result_intersect)
                for perm in permutations(parts):
                    current_set = set(indices)
                    skip = False
                    for part in perm:
                        rp = ratios.get(part)
                        if not isinstance(rp, dict):
                            skip = True
                            break
                        current_set = set(sorted(current_set, key=lambda x: rp.get(x, 1.0))[:rem3])
                        if len(current_set) == 0:
                            break
                    if not skip:
                        result.update(current_set)
                return list(result)

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
    # Always re-read the WSCS score from the precomputed array to avoid cross-level lookup issues.
    best_score = float(score_base_arr[anchor_idx])

    # Log anchor geometry: shoulder and wrist lines.
    anchor_c = batch_pose_data[anchor_idx]['bodies']['candidate']
    if has_pt(anchor_c[2]) and has_pt(anchor_c[5]):
        sh_dy = abs(float(anchor_c[5][1] - anchor_c[2][1]))
        sh_dx = abs(float(anchor_c[5][0] - anchor_c[2][0]))
        logger(f"[Anchor] Shoulder line: dy={sh_dy:.2f}px, angle={math.degrees(math.atan2(sh_dy, max(sh_dx, 1e-6))):.1f}°")
    else:
        logger("[Anchor] Shoulder line: missing keypoints")
    if has_pt(anchor_c[4]) and has_pt(anchor_c[7]):
        wr_dy = abs(float(anchor_c[7][1] - anchor_c[4][1]))
        wr_dx = abs(float(anchor_c[7][0] - anchor_c[4][0]))
        logger(f"[Anchor] Wrist line: dy={wr_dy:.2f}px, angle={math.degrees(math.atan2(wr_dy, max(wr_dx, 1e-6))):.1f}°")
    else:
        logger("[Anchor] Wrist line: missing keypoints")
    if has_pt(anchor_c[0]) and has_pt(anchor_c[16]) and has_pt(anchor_c[17]):
        d_r = math.sqrt((float(anchor_c[16][0]) - float(anchor_c[0][0])) ** 2 + (float(anchor_c[16][1]) - float(anchor_c[0][1])) ** 2)
        d_l = math.sqrt((float(anchor_c[17][0]) - float(anchor_c[0][0])) ** 2 + (float(anchor_c[17][1]) - float(anchor_c[0][1])) ** 2)
        ratio = max(d_r, d_l) / max(min(d_r, d_l), 1e-6)
        logger(f"[Anchor] Ear-nose ratio: {ratio:.3f} (right={d_r:.2f}px, left={d_l:.2f}px)")
    else:
        logger("[Anchor] Ear-nose ratio: missing keypoints")

    return anchor_idx, best_score, found_perfect_frame, found_degraded_frame, level1_scores, level2_scores
