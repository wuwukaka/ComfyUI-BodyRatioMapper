import numpy as np
from . import matrix_ops as matrix_ops_external

"""
Frame-level transformation operators extracted from ultimate_detector_node.

Design goals:
- Keep per-frame math and execution semantics unchanged.
- Accept dependencies (has_pt, safe_add, scale callbacks) via parameters to avoid hidden state.
- Preserve in-place mutation behavior for frame_data tensors.
"""


def build_stabilized_offset(base_offset, frame_idx, enable_final_offset_alignment, enable_offset_stabilizer, final_stabilizer_comps):
    """
    Build the final per-frame global offset.

    Rule:
    - Base alignment offset is applied only when enable_final_offset_alignment is True.
    - Stabilizer compensation is added on top when enabled and index is valid.
    """
    offset = base_offset.copy() if enable_final_offset_alignment else np.array([0.0, 0.0])
    if enable_offset_stabilizer and 0 <= frame_idx < len(final_stabilizer_comps):
        offset += final_stabilizer_comps[frame_idx]
    return offset


def force_align_face_hands_to_body(frame_data, has_pt, safe_add):
    """
    Force face and hands to follow body anchors.

    - Face anchor: body nose (candidate[0]) to face landmark 30.
    - Hand anchors: body wrists (candidate[7], candidate[4]) to hand root point 0.
    """
    faces = frame_data['faces']
    candidate = frame_data['bodies']['candidate']

    if isinstance(faces, np.ndarray) and faces.shape[1] > 30 and has_pt(faces):
        body_nose = candidate[0]
        if has_pt(body_nose):
            face_nose = faces[0, 30]
            if has_pt(face_nose):
                safe_add(faces, body_nose - face_nose)

    hands = frame_data['hands']
    if isinstance(hands, np.ndarray) and hands.shape[0] > 0:
        if hands.shape[0] > 0 and has_pt(hands[0]) and len(candidate) > 7:
            body_wr_l = candidate[7]
            if has_pt(body_wr_l):
                hand_wr_l = hands[0, 0]
                if has_pt(hand_wr_l):
                    safe_add(hands[0], body_wr_l - hand_wr_l)

        if hands.shape[0] > 1 and has_pt(hands[1]) and len(candidate) > 4:
            body_wr_r = candidate[4]
            if has_pt(body_wr_r):
                hand_wr_r = hands[1, 0]
                if has_pt(hand_wr_r):
                    safe_add(hands[1], body_wr_r - hand_wr_r)


def apply_global_offset_to_frame(frame_data, offset, has_pt, safe_add):
    """
    Apply one global translation vector to body/face/hands/feet in-place.

    Body uses point-wise has_pt checks.
    Face/hands/feet use safe_add to preserve existing masking behavior.
    """
    candidate = frame_data['bodies']['candidate']
    for idx in range(len(candidate)):
        if has_pt(candidate[idx]):
            candidate[idx] += offset

    safe_add(frame_data['faces'], offset)
    for hand_idx in range(2):
        safe_add(frame_data['hands'][hand_idx], offset)
    for foot_idx in range(2):
        safe_add(frame_data['feet'][foot_idx], offset)


def apply_neck_and_shoulder_offsets(frame_data, neck_offset, right_shoulder_offset, left_shoulder_offset, has_pt, safe_add):
    """
    Apply neck and shoulder chain offsets.

    - Neck offset affects body nose (index 0).
    - Right shoulder offset affects body indices (2,3,4) and right hand chain.
    - Left shoulder offset affects body indices (5,6,7) and left hand chain.
    """
    candidate = frame_data['bodies']['candidate']

    if has_pt(candidate[0]):
        candidate[0] += neck_offset

    for idx in (2, 3, 4):
        if has_pt(candidate[idx]):
            candidate[idx] += right_shoulder_offset
    safe_add(frame_data['hands'][1], right_shoulder_offset)

    for idx in (5, 6, 7):
        if has_pt(candidate[idx]):
            candidate[idx] += left_shoulder_offset
    safe_add(frame_data['hands'][0], left_shoulder_offset)


def apply_arm_chain_offsets(frame_data, raw_candidate, upper_arm_ratio, lower_arm_ratio, has_pt, safe_add):
    """
    Apply FK-style arm push along both arm chains.

    The function computes local segment offsets from current joint geometry:
    - Upper arm ratio controls shoulder->elbow segment push.
    - Lower arm ratio controls elbow->wrist segment push.
    Offsets are propagated to corresponding hand arrays.
    """
    candidate = frame_data['bodies']['candidate']

    # Right arm: 2 -> 3 -> 4, propagate same delta to right hand (index 1).
    if (
        isinstance(raw_candidate, np.ndarray)
        and len(raw_candidate) > 4
        and has_pt(candidate[2])
        and has_pt(candidate[3])
        and has_pt(raw_candidate[2])
        and has_pt(raw_candidate[3])
    ):
        raw_vec_23 = raw_candidate[3] - raw_candidate[2]
        new_3 = candidate[2] + raw_vec_23 * upper_arm_ratio
        delta_3 = new_3 - candidate[3]
        for idx in (3, 4):
            if has_pt(candidate[idx]):
                candidate[idx] += delta_3
        safe_add(frame_data['hands'][1], delta_3)

    if (
        isinstance(raw_candidate, np.ndarray)
        and len(raw_candidate) > 4
        and has_pt(candidate[3])
        and has_pt(candidate[4])
        and has_pt(raw_candidate[3])
        and has_pt(raw_candidate[4])
    ):
        raw_vec_34 = raw_candidate[4] - raw_candidate[3]
        new_4 = candidate[3] + raw_vec_34 * lower_arm_ratio
        delta_4 = new_4 - candidate[4]
        if has_pt(candidate[4]):
            candidate[4] += delta_4
        safe_add(frame_data['hands'][1], delta_4)

    # Left arm: 5 -> 6 -> 7, propagate same delta to left hand (index 0).
    if (
        isinstance(raw_candidate, np.ndarray)
        and len(raw_candidate) > 7
        and has_pt(candidate[5])
        and has_pt(candidate[6])
        and has_pt(raw_candidate[5])
        and has_pt(raw_candidate[6])
    ):
        raw_vec_56 = raw_candidate[6] - raw_candidate[5]
        new_6 = candidate[5] + raw_vec_56 * upper_arm_ratio
        delta_6 = new_6 - candidate[6]
        for idx in (6, 7):
            if has_pt(candidate[idx]):
                candidate[idx] += delta_6
        safe_add(frame_data['hands'][0], delta_6)

    if (
        isinstance(raw_candidate, np.ndarray)
        and len(raw_candidate) > 7
        and has_pt(candidate[6])
        and has_pt(candidate[7])
        and has_pt(raw_candidate[6])
        and has_pt(raw_candidate[7])
    ):
        raw_vec_67 = raw_candidate[7] - raw_candidate[6]
        new_7 = candidate[6] + raw_vec_67 * lower_arm_ratio
        delta_7 = new_7 - candidate[7]
        if has_pt(candidate[7]):
            candidate[7] += delta_7
        safe_add(frame_data['hands'][0], delta_7)


def apply_leg_chain_offsets(frame_data, raw_candidate, hip_width_scale, ll1_ratio, ll2_ratio, rl1_ratio, rl2_ratio, has_pt, safe_add):
    """
    Apply hip-width scaling and subsequent leg-chain pushing.

    Execution order (must stay fixed):
    1) Scale hips around hip midpoint (indices 8 and 11).
    2) Propagate hip translation to downstream knees/ankles/feet.
    3) Apply thigh and calf offsets for both sides.
    """
    candidate = frame_data['bodies']['candidate']

    if has_pt(candidate[8]) and has_pt(candidate[11]):
        hip_r_old = candidate[8].copy()
        hip_l_old = candidate[11].copy()
        hip_mid = (hip_r_old + hip_l_old) * 0.5
        candidate[8] = hip_mid + (hip_r_old - hip_mid) * hip_width_scale
        candidate[11] = hip_mid + (hip_l_old - hip_mid) * hip_width_scale
        hip_r_delta = candidate[8] - hip_r_old
        hip_l_delta = candidate[11] - hip_l_old

        for idx in (9, 10):
            if has_pt(candidate[idx]):
                candidate[idx] += hip_r_delta
        for idx in (12, 13):
            if has_pt(candidate[idx]):
                candidate[idx] += hip_l_delta
        safe_add(frame_data['feet'][1], hip_r_delta)
        safe_add(frame_data['feet'][0], hip_l_delta)

    # Right leg in this index convention: 8 -> 9 -> 10, propagate to right foot (index 1).
    if (
        isinstance(raw_candidate, np.ndarray)
        and len(raw_candidate) > 10
        and has_pt(candidate[8])
        and has_pt(candidate[9])
        and has_pt(raw_candidate[8])
        and has_pt(raw_candidate[9])
    ):
        raw_vec_89 = raw_candidate[9] - raw_candidate[8]
        new_9 = candidate[8] + raw_vec_89 * ll1_ratio
        delta_9 = new_9 - candidate[9]
        for idx in (9, 10):
            if has_pt(candidate[idx]):
                candidate[idx] += delta_9
        safe_add(frame_data['feet'][1], delta_9)

    if (
        isinstance(raw_candidate, np.ndarray)
        and len(raw_candidate) > 10
        and has_pt(candidate[9])
        and has_pt(candidate[10])
        and has_pt(raw_candidate[9])
        and has_pt(raw_candidate[10])
    ):
        raw_vec_910 = raw_candidate[10] - raw_candidate[9]
        new_10 = candidate[9] + raw_vec_910 * ll2_ratio
        delta_10 = new_10 - candidate[10]
        if has_pt(candidate[10]):
            candidate[10] += delta_10
        safe_add(frame_data['feet'][1], delta_10)

    # Left leg in this index convention: 11 -> 12 -> 13, propagate to left foot (index 0).
    if (
        isinstance(raw_candidate, np.ndarray)
        and len(raw_candidate) > 13
        and has_pt(candidate[11])
        and has_pt(candidate[12])
        and has_pt(raw_candidate[11])
        and has_pt(raw_candidate[12])
    ):
        raw_vec_1112 = raw_candidate[12] - raw_candidate[11]
        new_12 = candidate[11] + raw_vec_1112 * rl1_ratio
        delta_12 = new_12 - candidate[12]
        for idx in (12, 13):
            if has_pt(candidate[idx]):
                candidate[idx] += delta_12
        safe_add(frame_data['feet'][0], delta_12)

    if (
        isinstance(raw_candidate, np.ndarray)
        and len(raw_candidate) > 13
        and has_pt(candidate[12])
        and has_pt(candidate[13])
        and has_pt(raw_candidate[12])
        and has_pt(raw_candidate[13])
    ):
        raw_vec_1213 = raw_candidate[13] - raw_candidate[12]
        new_13 = candidate[12] + raw_vec_1213 * rl2_ratio
        delta_13 = new_13 - candidate[13]
        if has_pt(candidate[13]):
            candidate[13] += delta_13
        safe_add(frame_data['feet'][0], delta_13)


def apply_rigid_head_points(candidate, original_candidate, face_x_scale, face_y_scale, eye_width_scale, has_pt):
    """
    Rebuild rigid head points (14/15/16/17) around the translated nose anchor.

    The function keeps original relative vectors from nose but scales X/Y independently.
    """
    orig_nose = original_candidate[0]
    new_nose = candidate[0]
    if has_pt(orig_nose) and has_pt(new_nose):
        for idx in [14, 15, 16, 17]:
            if has_pt(original_candidate[idx]):
                dx = original_candidate[idx][0] - orig_nose[0]
                dy = original_candidate[idx][1] - orig_nose[1]
                x_scale = eye_width_scale if idx in [14, 15] else face_x_scale
                candidate[idx][0] = new_nose[0] + dx * x_scale
                candidate[idx][1] = new_nose[1] + dy * face_y_scale


def apply_face_rigid_mask(faces, original_faces, face_x_scale, face_y_scale, safe_add):
    """
    Apply rigid-mask scaling to face landmarks.

    Preferred mode:
    - Center at face landmark 30, scale in local coordinates, then add delta via safe_add.

    Fallback mode:
    - Direct XY scaling when landmark 30 is unavailable in shape.
    """
    if faces.shape[1] > 30:
        face_nose_tip_idx = 30
        current_face_nose = original_faces[0, face_nose_tip_idx, :].copy()
        faces_new = matrix_ops_external.scale_about_center(
            original_faces,
            current_face_nose[np.newaxis, np.newaxis, :],
            face_x_scale,
            face_y_scale
        )
        delta = faces_new - original_faces
        safe_add(faces, delta[0])
    else:
        faces[:, :, 0] *= face_x_scale
        faces[:, :, 1] *= face_y_scale


def try_apply_face_rigid_mask(frame_data, original_faces, face_x_scale, face_y_scale, has_pt, safe_add, require_nonzero_sum=False):
    """
    Conditionally apply face rigid mask under caller-defined trigger policy.

    - require_nonzero_sum=True: gate by aggregate nonzero magnitude.
    - otherwise: gate by standard has_pt check on face tensor.
    """
    faces = frame_data['faces']
    if require_nonzero_sum:
        if np.sum(np.abs(faces)) > 0.01:
            apply_face_rigid_mask(faces, original_faces, face_x_scale, face_y_scale, safe_add)
        return
    if len(faces) > 0 and has_pt(faces):
        apply_face_rigid_mask(faces, original_faces, face_x_scale, face_y_scale, safe_add)


def apply_face_mask_for_frame(frame_data, face_x_scale, face_y_scale, eye_width_scale, has_pt, safe_add, require_nonzero_sum=False):
    """
    Snapshot current face tensor and apply rigid-mask transform using that snapshot.

    This preserves original per-frame behavior where face deltas are based on pre-update data.
    """
    original_faces_local = frame_data['faces'].copy()
    try_apply_face_rigid_mask(
        frame_data,
        original_faces_local,
        face_x_scale,
        face_y_scale,
        has_pt,
        safe_add,
        require_nonzero_sum=require_nonzero_sum
    )

    # Apply eye-region-specific X adjustment after global face scaling.
    if abs(face_x_scale) > 1e-6:
        eye_adjust_ratio = eye_width_scale / face_x_scale
        if abs(eye_adjust_ratio - 1.0) > 1e-6:
            faces = frame_data['faces']
            if isinstance(faces, np.ndarray) and faces.shape[0] > 0 and faces.shape[1] >= 48:
                for eye_indices in ([36, 37, 38, 39, 40, 41], [42, 43, 44, 45, 46, 47]):
                    pts = []
                    for idx in eye_indices:
                        pt = faces[0, idx]
                        if has_pt(pt):
                            pts.append(pt)
                    if len(pts) >= 2:
                        eye_center = np.mean(np.stack(pts, axis=0), axis=0)
                        for idx in eye_indices:
                            if has_pt(faces[0, idx]):
                                dx = faces[0, idx, 0] - eye_center[0]
                                faces[0, idx, 0] = eye_center[0] + dx * eye_adjust_ratio


def apply_spine_offset_to_lower_body(frame_data, spine_offset, has_pt, safe_add):
    """
    Apply spine offset to lower-body chain and feet.

    Body indices affected: 8,9,10,11,12,13.
    The same offset is propagated to both feet arrays.
    """
    candidate = frame_data['bodies']['candidate']
    for idx in [8, 9, 10, 11, 12, 13]:
        if has_pt(candidate[idx]):
            candidate[idx] += spine_offset
    safe_add(frame_data['feet'][0], spine_offset)
    safe_add(frame_data['feet'][1], spine_offset)


def scale_frame_to_canvas(frame_data, x_ratio, y_ratio, has_pt, use_body_guard=False):
    """
    Scale frame coordinates by canvas XY ratios.

    - Body scaling can be guarded by use_body_guard + has_pt.
    - Hands and feet are always scaled to keep chain consistency.
    """
    if (not use_body_guard) or has_pt(frame_data['bodies']['candidate']):
        frame_data['bodies']['candidate'][:, 0] *= x_ratio
        frame_data['bodies']['candidate'][:, 1] *= y_ratio
    frame_data['hands'][:, :, 0] *= x_ratio
    frame_data['hands'][:, :, 1] *= y_ratio
    frame_data['feet'][:, :, 0] *= x_ratio
    frame_data['feet'][:, :, 1] *= y_ratio


def apply_extremity_scaling(frame_data, candidate_points, true_hand_scale, true_foot_edge1_scale, true_foot_edge2_scale, true_foot_edge3_scale, true_body_to_foot_ankle_scale, enable_foot_internal_scaling, scale_hand_isotropically, scale_foot_by_edges):
    """
    Apply hand and foot internal scaling before/after body-chain stages (as directed by caller).

    - Hands are always scaled isotropically around wrist pivots.
    - Feet internal scaling is optional via enable_foot_internal_scaling.
    """
    scale_hand_isotropically(frame_data['hands'][0], true_hand_scale, candidate_points[7])
    scale_hand_isotropically(frame_data['hands'][1], true_hand_scale, candidate_points[4])

    if enable_foot_internal_scaling:
        scale_foot_by_edges(frame_data['feet'][0], true_foot_edge1_scale, true_foot_edge2_scale, true_foot_edge3_scale, true_body_to_foot_ankle_scale, candidate_points[13])
        scale_foot_by_edges(frame_data['feet'][1], true_foot_edge1_scale, true_foot_edge2_scale, true_foot_edge3_scale, true_body_to_foot_ankle_scale, candidate_points[10])
