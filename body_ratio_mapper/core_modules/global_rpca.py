import numpy as np


def calculate_13_bone_global_rpca(anchor_candidate, anchor_faces, f0_candidate, f0_faces, alignment_mode=True):
    """
    Compute the global RPCA multiplier from anchor frame vs frame-0 geometry.

    Design intent:
    - Use a fixed set of 13 structural measurements (11 body segments + ear width + face height).
    - Convert each measurement into an anchor/f0 ratio.
    - Rank ratios by absolute deviation from 1.0 and pick a robust representative:
      use the 3rd closest ratio when enough samples exist, otherwise use the closest available fallback.

    Notes:
    - This function is intentionally deterministic and independent of per-frame scaling loops.
    - When alignment_mode is disabled, the multiplier is forced to 1.0.
    """
    if not alignment_mode:
        return 1.0

    # Internal "point exists" rule follows the project-wide non-zero threshold.
    def has_pt(pt):
        return np.sum(np.abs(pt)) > 0.01

    # Distance helper that returns None when either endpoint is unavailable.
    def get_physical_dist(p1, p2):
        if not has_pt(p1) or not has_pt(p2):
            return None
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Stores tuples: (ratio, |ratio-1|, semantic_name)
    bone_ratios = []

    # Append one segment ratio if both anchor/f0 segment lengths are valid.
    def add_bone_ratio(idx1, idx2, anc_c, f0_c, name):
        anc_len = get_physical_dist(anc_c[idx1], anc_c[idx2])
        f0_len = get_physical_dist(f0_c[idx1], f0_c[idx2])
        if anc_len and f0_len and f0_len > 1e-6:
            ratio = anc_len / f0_len
            deviation = abs(ratio - 1.0)
            bone_ratios.append((ratio, deviation, name))

    # 11 core body measurements.
    add_bone_ratio(1, 11, anchor_candidate, f0_candidate, "left_torso")
    add_bone_ratio(1, 8, anchor_candidate, f0_candidate, "right_torso")
    add_bone_ratio(2, 5, anchor_candidate, f0_candidate, "shoulder_width")
    add_bone_ratio(5, 6, anchor_candidate, f0_candidate, "left_upper_arm")
    add_bone_ratio(2, 3, anchor_candidate, f0_candidate, "right_upper_arm")
    add_bone_ratio(6, 7, anchor_candidate, f0_candidate, "left_lower_arm")
    add_bone_ratio(3, 4, anchor_candidate, f0_candidate, "right_lower_arm")
    add_bone_ratio(11, 12, anchor_candidate, f0_candidate, "left_upper_leg")
    add_bone_ratio(8, 9, anchor_candidate, f0_candidate, "right_upper_leg")
    add_bone_ratio(12, 13, anchor_candidate, f0_candidate, "left_lower_leg")
    add_bone_ratio(9, 10, anchor_candidate, f0_candidate, "right_lower_leg")

    # 12th measurement: ear width.
    anc_ear = get_physical_dist(anchor_candidate[16], anchor_candidate[17])
    f0_ear = get_physical_dist(f0_candidate[16], f0_candidate[17])
    if anc_ear and f0_ear and f0_ear > 1e-6:
        ratio = anc_ear / f0_ear
        bone_ratios.append((ratio, abs(ratio - 1.0), "ear_width"))

    # 13th measurement: face height (chin to glabella-like landmark in current index convention).
    if len(anchor_faces) > 0 and len(f0_faces) > 0:
        anc_face = anchor_faces[0]
        f0_face = f0_faces[0]
        if len(anc_face) > 27 and len(f0_face) > 27:
            anc_face_h = get_physical_dist(anc_face[8], anc_face[27])
            f0_face_h = get_physical_dist(f0_face[8], f0_face[27])
            if anc_face_h and f0_face_h and f0_face_h > 1e-6:
                ratio = anc_face_h / f0_face_h
                bone_ratios.append((ratio, abs(ratio - 1.0), "face_height"))

    # Robust representative selection:
    # - >=3 ratios: use the 3rd nearest-to-1.0 ratio to reduce sensitivity to local outliers.
    # - 1..2 ratios: use the farthest available among the near set (existing historical behavior).
    # - 0 ratios: fallback to neutral multiplier.
    if len(bone_ratios) >= 3:
        bone_ratios.sort(key=lambda x: x[1])
        return bone_ratios[2][0]
    if len(bone_ratios) > 0:
        bone_ratios.sort(key=lambda x: x[1])
        return bone_ratios[-1][0]
    return 1.0


def forge_final_scale_constants(fk_values, global_rpca_multiplier, alignment_mode=True):
    """
    Build final per-part scales from FK values and global RPCA multiplier.

    Rule:
    - alignment_mode=True  -> final_scale = fk * global_rpca_multiplier
    - alignment_mode=False -> final_scale = fk * 1.0

    Output keys are kept stable for downstream compatibility.
    """
    final_scales = {}
    multiplier = global_rpca_multiplier if alignment_mode else 1.0

    # Core body segments.
    final_scales['torso'] = fk_values['torso'] * multiplier
    final_scales['neck'] = fk_values['neck'] * multiplier
    final_scales['shoulder'] = fk_values['shoulder'] * multiplier
    final_scales['hip_width'] = fk_values['hip_width'] * multiplier
    final_scales['upper_arm'] = fk_values['upper_arm'] * multiplier
    final_scales['lower_arm'] = fk_values['lower_arm'] * multiplier
    final_scales['upper_leg'] = fk_values['upper_leg'] * multiplier
    final_scales['lower_leg'] = fk_values['lower_leg'] * multiplier

    # Extremities and fine structure scales.
    final_scales['hand'] = fk_values['hand'] * multiplier
    final_scales['foot_edge1'] = fk_values['foot_edge1'] * multiplier
    final_scales['foot_edge2'] = fk_values['foot_edge2'] * multiplier
    final_scales['foot_edge3'] = fk_values['foot_edge3'] * multiplier
    final_scales['body_to_foot_ankle'] = fk_values['body_to_foot_ankle'] * multiplier

    # Face scales are split on X/Y by design.
    final_scales['face_x'] = fk_values['face_x'] * multiplier
    final_scales['face_y'] = fk_values['face_y'] * multiplier
    final_scales['eye_width'] = fk_values['eye_width'] * multiplier

    return final_scales
