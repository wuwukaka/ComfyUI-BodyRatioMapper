import numpy as np

"""
Scale/FK solver utilities extracted from the main node.

Scope:
- Left-right complement helpers.
- FK extraction for torso/neck, arms, legs, hands/feet/face.
- Hand FK protection against outlier amplification.

Design constraints:
- Keep legacy thresholds and fallback order unchanged.
- Keep return shapes stable for drop-in compatibility.
"""


def lr_complement(left, right):
    """Mirror-complement two side lengths: fill missing side with the available one."""
    left_eff = left if left is not None else right
    right_eff = right if right is not None else left
    return left_eff, right_eff


def avg_ratio_two_sides(ref_l, anc_l, ref_r, anc_r):
    """
    Compute mean FK ratio over valid left/right pairs.

    Uses ref/anchor ratio per side and averages available sides.
    Returns 1.0 when neither side is valid.
    """
    ratios = []
    if ref_l is not None and anc_l is not None and ref_l > 1e-6 and anc_l > 1e-6:
        ratios.append(ref_l / anc_l)
    if ref_r is not None and anc_r is not None and ref_r > 1e-6 and anc_r > 1e-6:
        ratios.append(ref_r / anc_r)
    return (sum(ratios) / len(ratios)) if ratios else 1.0


def extract_fk_values_part1_torso_neck(ref_candidate, anc_candidate):
    """
    Extract FK for torso, neck, shoulder, and hip_width.

    Rules:
    - Torso uses left-right complement before averaging.
    - Neck and shoulder use direct ratios when valid.
    - hip_width is tied to shoulder FK by design.
    """
    def has_pt(pt):
        return np.sum(np.abs(pt)) > 0.01

    def get_physical_dist(p1, p2):
        if not has_pt(p1) or not has_pt(p2):
            return None
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    ref_torso_l = get_physical_dist(ref_candidate[1], ref_candidate[11])
    anc_torso_l = get_physical_dist(anc_candidate[1], anc_candidate[11])
    ref_torso_r = get_physical_dist(ref_candidate[1], ref_candidate[8])
    anc_torso_r = get_physical_dist(anc_candidate[1], anc_candidate[8])

    ref_torso_l_eff, ref_torso_r_eff = lr_complement(ref_torso_l, ref_torso_r)
    anc_torso_l_eff, anc_torso_r_eff = lr_complement(anc_torso_l, anc_torso_r)
    fk_torso = avg_ratio_two_sides(ref_torso_l_eff, anc_torso_l_eff, ref_torso_r_eff, anc_torso_r_eff)

    ref_neck = get_physical_dist(ref_candidate[0], ref_candidate[1])
    anc_neck = get_physical_dist(anc_candidate[0], anc_candidate[1])
    fk_neck = (ref_neck / anc_neck) if (ref_neck and anc_neck and anc_neck > 1e-6) else 1.0

    ref_shoulder = get_physical_dist(ref_candidate[2], ref_candidate[5])
    anc_shoulder = get_physical_dist(anc_candidate[2], anc_candidate[5])
    fk_shoulder = (ref_shoulder / anc_shoulder) if (ref_shoulder and anc_shoulder and anc_shoulder > 1e-6) else 1.0

    fk_hip_width = fk_shoulder
    return fk_torso, fk_neck, fk_shoulder, fk_hip_width


def extract_fk_values_part2_arms(ref_candidate, anc_candidate):
    """
    Extract upper/lower arm FK with fallback and mirror compensation.

    Strategy:
    - If both elbows are unavailable, degrade to full-arm FK and return same FK for both segments.
    - Otherwise compute upper/lower separately with left-right complement.
    - If reference upper/lower gap exceeds 1.3x, lock both computations to the longer reference baseline.
    - If reference lower arm is missing on both sides, lower_arm FK inherits upper_arm FK.
    """
    def has_pt(pt):
        return np.sum(np.abs(pt)) > 0.01

    def get_physical_dist(p1, p2):
        if not has_pt(p1) or not has_pt(p2):
            return None
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    has_left_elbow = has_pt(ref_candidate[6]) and has_pt(anc_candidate[6])
    has_right_elbow = has_pt(ref_candidate[3]) and has_pt(anc_candidate[3])

    if not has_left_elbow and not has_right_elbow:
        ref_full_l = get_physical_dist(ref_candidate[5], ref_candidate[7])
        anc_full_l = get_physical_dist(anc_candidate[5], anc_candidate[7])
        ref_full_r = get_physical_dist(ref_candidate[2], ref_candidate[4])
        anc_full_r = get_physical_dist(anc_candidate[2], anc_candidate[4])
        ref_full_l_eff, ref_full_r_eff = lr_complement(ref_full_l, ref_full_r)
        anc_full_l_eff, anc_full_r_eff = lr_complement(anc_full_l, anc_full_r)
        fk_arm = avg_ratio_two_sides(ref_full_l_eff, anc_full_l_eff, ref_full_r_eff, anc_full_r_eff)
        return fk_arm, fk_arm, False

    ref_upper_l = get_physical_dist(ref_candidate[5], ref_candidate[6])
    anc_upper_l = get_physical_dist(anc_candidate[5], anc_candidate[6])
    ref_upper_r = get_physical_dist(ref_candidate[2], ref_candidate[3])
    anc_upper_r = get_physical_dist(anc_candidate[2], anc_candidate[3])
    ref_lower_l = get_physical_dist(ref_candidate[6], ref_candidate[7])
    anc_lower_l = get_physical_dist(anc_candidate[6], anc_candidate[7])
    ref_lower_r = get_physical_dist(ref_candidate[3], ref_candidate[4])
    anc_lower_r = get_physical_dist(anc_candidate[3], anc_candidate[4])

    ref_upper_l_eff, ref_upper_r_eff = lr_complement(ref_upper_l, ref_upper_r)
    ref_lower_l_eff, ref_lower_r_eff = lr_complement(ref_lower_l, ref_lower_r)
    anc_upper_l_eff, anc_upper_r_eff = lr_complement(anc_upper_l, anc_upper_r)
    anc_lower_l_eff, anc_lower_r_eff = lr_complement(anc_lower_l, anc_lower_r)

    ref_upper_vals = [x for x in [ref_upper_l_eff, ref_upper_r_eff] if x is not None and x > 1e-6]
    ref_lower_vals = [x for x in [ref_lower_l_eff, ref_lower_r_eff] if x is not None and x > 1e-6]
    arm_ref_lock_to_long = False
    shared_ref_arm_baseline = None
    if len(ref_upper_vals) > 0 and len(ref_lower_vals) > 0:
        ref_upper_mean = sum(ref_upper_vals) / len(ref_upper_vals)
        ref_lower_mean = sum(ref_lower_vals) / len(ref_lower_vals)
        ref_max = max(ref_upper_mean, ref_lower_mean)
        ref_min = min(ref_upper_mean, ref_lower_mean)
        if ref_max > ref_min * 1.3:
            arm_ref_lock_to_long = True
            shared_ref_arm_baseline = ref_max

    if arm_ref_lock_to_long:
        fk_upper_arm = avg_ratio_two_sides(shared_ref_arm_baseline, anc_upper_l_eff, shared_ref_arm_baseline, anc_upper_r_eff)
    else:
        fk_upper_arm = avg_ratio_two_sides(ref_upper_l_eff, anc_upper_l_eff, ref_upper_r_eff, anc_upper_r_eff)

    if arm_ref_lock_to_long:
        fk_lower_arm_calc = avg_ratio_two_sides(shared_ref_arm_baseline, anc_lower_l_eff, shared_ref_arm_baseline, anc_lower_r_eff)
    else:
        fk_lower_arm_calc = avg_ratio_two_sides(ref_lower_l_eff, anc_lower_l_eff, ref_lower_r_eff, anc_lower_r_eff)

    fk_lower_arm = fk_upper_arm if (ref_lower_l is None and ref_lower_r is None) else fk_lower_arm_calc
    return fk_upper_arm, fk_lower_arm, arm_ref_lock_to_long


def extract_fk_values_part3_legs(ref_candidate, anc_candidate):
    """
    Extract upper/lower leg FK with fallback and mirror compensation.

    Strategy:
    - If both knees are unavailable, degrade to full-leg FK and apply same FK to both segments.
    - Otherwise compute upper/lower legs separately with left-right complement.
    - If reference lower leg is missing on both sides, lower_leg FK inherits upper_leg FK.
    """
    def has_pt(pt):
        return np.sum(np.abs(pt)) > 0.01

    def get_physical_dist(p1, p2):
        if not has_pt(p1) or not has_pt(p2):
            return None
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    has_left_knee = has_pt(ref_candidate[12]) and has_pt(anc_candidate[12])
    has_right_knee = has_pt(ref_candidate[9]) and has_pt(anc_candidate[9])

    if not has_left_knee and not has_right_knee:
        ref_full_l = get_physical_dist(ref_candidate[11], ref_candidate[13])
        anc_full_l = get_physical_dist(anc_candidate[11], anc_candidate[13])
        ref_full_r = get_physical_dist(ref_candidate[8], ref_candidate[10])
        anc_full_r = get_physical_dist(anc_candidate[8], anc_candidate[10])
        ref_full_l_eff, ref_full_r_eff = lr_complement(ref_full_l, ref_full_r)
        anc_full_l_eff, anc_full_r_eff = lr_complement(anc_full_l, anc_full_r)
        fk_leg = avg_ratio_two_sides(ref_full_l_eff, anc_full_l_eff, ref_full_r_eff, anc_full_r_eff)
        return fk_leg, fk_leg

    ref_upper_l = get_physical_dist(ref_candidate[11], ref_candidate[12])
    anc_upper_l = get_physical_dist(anc_candidate[11], anc_candidate[12])
    ref_upper_r = get_physical_dist(ref_candidate[8], ref_candidate[9])
    anc_upper_r = get_physical_dist(anc_candidate[8], anc_candidate[9])
    ref_upper_l_eff, ref_upper_r_eff = lr_complement(ref_upper_l, ref_upper_r)
    anc_upper_l_eff, anc_upper_r_eff = lr_complement(anc_upper_l, anc_upper_r)
    fk_upper_leg = avg_ratio_two_sides(ref_upper_l_eff, anc_upper_l_eff, ref_upper_r_eff, anc_upper_r_eff)

    ref_lower_l = get_physical_dist(ref_candidate[12], ref_candidate[13])
    anc_lower_l = get_physical_dist(anc_candidate[12], anc_candidate[13])
    ref_lower_r = get_physical_dist(ref_candidate[9], ref_candidate[10])
    anc_lower_r = get_physical_dist(anc_candidate[9], anc_candidate[10])
    ref_lower_l_eff, ref_lower_r_eff = lr_complement(ref_lower_l, ref_lower_r)
    anc_lower_l_eff, anc_lower_r_eff = lr_complement(anc_lower_l, anc_lower_r)
    fk_lower_leg_calc = avg_ratio_two_sides(ref_lower_l_eff, anc_lower_l_eff, ref_lower_r_eff, anc_lower_r_eff)

    fk_lower_leg = fk_upper_leg if (ref_lower_l is None and ref_lower_r is None) else fk_lower_leg_calc
    return fk_upper_leg, fk_lower_leg


def extract_fk_values_part4_hands_feet_face(ref_candidate, ref_faces, ref_hands, ref_feet, anc_candidate, anc_faces, anc_hands, anc_feet, anc_hand_baseline=None):
    """
    Extract FK for hands, feet, body-to-foot-ankle distance, and face X/Y scales.

    Hand:
    - Uses topology path length (0->9->10->11->12), with left-right complement.
    - If anc_hand_baseline is provided, reference side lengths are compared to that baseline.
      Otherwise compare side-by-side ref/anchor lengths.

    Foot:
    - Computes three independent edges per foot (ankle-bigtoe, ankle-smalltoe, bigtoe-smalltoe).
    - Applies left-right complement per edge, then averages side ratios.

    Body-to-foot:
    - Uses ankle (body) to foot-ankle distance with left-right complement.

    Face:
    - X scale from ear width.
    - Y scale from face landmark distance (8 to 27) when available.
    - Eye-width scale from body eye points (14 to 15).
    """
    def has_pt(pt):
        return np.sum(np.abs(pt)) > 0.01

    def get_physical_dist(p1, p2):
        if not has_pt(p1) or not has_pt(p2):
            return None
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Mid-finger chain length used as hand size proxy.
    def get_hand_topology_length(hand_keypoints):
        if len(hand_keypoints) == 0:
            return None
        hand = hand_keypoints[0]
        segments = [(0, 9), (9, 10), (10, 11), (11, 12)]
        total = 0
        for i, j in segments:
            if i < len(hand) and j < len(hand):
                seg = get_physical_dist(hand[i], hand[j])
                if seg is None:
                    return None
                total += seg
        return total if total > 0 else None

    ref_hand_l = get_hand_topology_length(ref_hands[:1])
    anc_hand_l = get_hand_topology_length(anc_hands[:1])
    ref_hand_r = get_hand_topology_length(ref_hands[1:2])
    anc_hand_r = get_hand_topology_length(anc_hands[1:2])
    ref_hand_l_eff, ref_hand_r_eff = lr_complement(ref_hand_l, ref_hand_r)
    anc_hand_l_eff, anc_hand_r_eff = lr_complement(anc_hand_l, anc_hand_r)

    hand_ratios = []
    if anc_hand_baseline is not None and anc_hand_baseline > 1e-6:
        if ref_hand_l_eff:
            hand_ratios.append(ref_hand_l_eff / anc_hand_baseline)
        if ref_hand_r_eff:
            hand_ratios.append(ref_hand_r_eff / anc_hand_baseline)
    else:
        if ref_hand_l_eff and anc_hand_l_eff and anc_hand_l_eff > 1e-6:
            hand_ratios.append(ref_hand_l_eff / anc_hand_l_eff)
        if ref_hand_r_eff and anc_hand_r_eff and anc_hand_r_eff > 1e-6:
            hand_ratios.append(ref_hand_r_eff / anc_hand_r_eff)
    fk_hand = sum(hand_ratios) / len(hand_ratios) if hand_ratios else 1.0

    # Three-edge foot model used by current internal foot scaling.
    def get_foot_edges(foot_keypoints):
        if len(foot_keypoints) == 0:
            return None, None, None
        foot = foot_keypoints[0]
        if len(foot) < 3:
            return None, None, None
        # Keep legacy internal foot-shape topology for edge FK:
        # root=idx0, toe_a=idx1, toe_b=idx2.
        root = foot[0]
        toe_a = foot[1]
        toe_b = foot[2]
        edge1 = get_physical_dist(root, toe_a)
        edge2 = get_physical_dist(root, toe_b)
        edge3 = get_physical_dist(toe_a, toe_b)
        return edge1, edge2, edge3

    ref_foot_l_e1, ref_foot_l_e2, ref_foot_l_e3 = get_foot_edges(ref_feet[:1])
    anc_foot_l_e1, anc_foot_l_e2, anc_foot_l_e3 = get_foot_edges(anc_feet[:1])
    ref_foot_r_e1, ref_foot_r_e2, ref_foot_r_e3 = get_foot_edges(ref_feet[1:2])
    anc_foot_r_e1, anc_foot_r_e2, anc_foot_r_e3 = get_foot_edges(anc_feet[1:2])

    ref_foot_l_e1_eff, ref_foot_r_e1_eff = lr_complement(ref_foot_l_e1, ref_foot_r_e1)
    anc_foot_l_e1_eff, anc_foot_r_e1_eff = lr_complement(anc_foot_l_e1, anc_foot_r_e1)
    ref_foot_l_e2_eff, ref_foot_r_e2_eff = lr_complement(ref_foot_l_e2, ref_foot_r_e2)
    anc_foot_l_e2_eff, anc_foot_r_e2_eff = lr_complement(anc_foot_l_e2, anc_foot_r_e2)
    ref_foot_l_e3_eff, ref_foot_r_e3_eff = lr_complement(ref_foot_l_e3, ref_foot_r_e3)
    anc_foot_l_e3_eff, anc_foot_r_e3_eff = lr_complement(anc_foot_l_e3, anc_foot_r_e3)

    fk_foot_edge1 = avg_ratio_two_sides(ref_foot_l_e1_eff, anc_foot_l_e1_eff, ref_foot_r_e1_eff, anc_foot_r_e1_eff)
    fk_foot_edge2 = avg_ratio_two_sides(ref_foot_l_e2_eff, anc_foot_l_e2_eff, ref_foot_r_e2_eff, anc_foot_r_e2_eff)
    fk_foot_edge3 = avg_ratio_two_sides(ref_foot_l_e3_eff, anc_foot_l_e3_eff, ref_foot_r_e3_eff, anc_foot_r_e3_eff)

    # Body-to-foot anchor uses heel (index 2) as foot root.
    ref_body_to_foot_left = get_physical_dist(ref_candidate[13], ref_feet[0][2]) if len(ref_feet[0]) > 2 and has_pt(ref_candidate[13]) and has_pt(ref_feet[0][2]) else None
    anc_body_to_foot_left = get_physical_dist(anc_candidate[13], anc_feet[0][2]) if len(anc_feet[0]) > 2 and has_pt(anc_candidate[13]) and has_pt(anc_feet[0][2]) else None
    ref_body_to_foot_right = get_physical_dist(ref_candidate[10], ref_feet[1][2]) if len(ref_feet[1]) > 2 and has_pt(ref_candidate[10]) and has_pt(ref_feet[1][2]) else None
    anc_body_to_foot_right = get_physical_dist(anc_candidate[10], anc_feet[1][2]) if len(anc_feet[1]) > 2 and has_pt(anc_candidate[10]) and has_pt(anc_feet[1][2]) else None
    ref_body_to_foot_left_eff, ref_body_to_foot_right_eff = lr_complement(ref_body_to_foot_left, ref_body_to_foot_right)
    anc_body_to_foot_left_eff, anc_body_to_foot_right_eff = lr_complement(anc_body_to_foot_left, anc_body_to_foot_right)
    fk_body_to_foot_ankle = avg_ratio_two_sides(
        ref_body_to_foot_left_eff, anc_body_to_foot_left_eff, ref_body_to_foot_right_eff, anc_body_to_foot_right_eff
    )

    ref_ear = get_physical_dist(ref_candidate[16], ref_candidate[17])
    anc_ear = get_physical_dist(anc_candidate[16], anc_candidate[17])
    fk_face_x = (ref_ear / anc_ear) if (ref_ear and anc_ear and anc_ear > 1e-6) else 1.0

    if len(ref_faces) > 0 and len(anc_faces) > 0:
        ref_face = ref_faces[0]
        anc_face = anc_faces[0]
        if len(ref_face) > 27 and len(anc_face) > 27:
            ref_face_h = get_physical_dist(ref_face[8], ref_face[27])
            anc_face_h = get_physical_dist(anc_face[8], anc_face[27])
            fk_face_y = (ref_face_h / anc_face_h) if (ref_face_h and anc_face_h and anc_face_h > 1e-6) else 1.0
        else:
            fk_face_y = 1.0
    else:
        fk_face_y = 1.0

    ref_eye = get_physical_dist(ref_candidate[14], ref_candidate[15])
    anc_eye = get_physical_dist(anc_candidate[14], anc_candidate[15])
    fk_eye_width = (ref_eye / anc_eye) if (ref_eye and anc_eye and anc_eye > 1e-6) else 1.0

    return fk_hand, fk_foot_edge1, fk_foot_edge2, fk_foot_edge3, fk_body_to_foot_ankle, fk_face_x, fk_face_y, fk_eye_width


def validate_hand_fk(hand_fk, torso, neck, upper_arm, lower_arm, upper_leg, lower_leg, foot_edge1, foot_edge2, foot_edge3, body_to_foot_ankle, face_x, face_y, eye_width=None, logger=print):
    """
    Protect hand FK from anomalous oversized values.

    Rule:
    - Collect non-neutral FK values from other body parts.
    - Clamp hand_fk to max_other_fk when hand_fk exceeds max_other_fk or exceeds 2.5.
    """
    other_fks = [torso, neck, upper_arm, lower_arm, upper_leg, lower_leg, foot_edge1, foot_edge2, foot_edge3, body_to_foot_ankle, face_x, face_y]
    if eye_width is not None:
        other_fks.append(eye_width)
    measured_fks = [fk for fk in other_fks if abs(fk - 1.0) > 0.01]
    max_other_fk = max(measured_fks) if measured_fks else 1.0

    if hand_fk > max_other_fk or hand_fk > 2.5:
        logger(f"[Hand FK Protection] Anomaly detected: hand_fk={hand_fk:.3f}, max_other={max_other_fk:.3f}. Clamping to max_other.")
        return max_other_fk
    return hand_fk
