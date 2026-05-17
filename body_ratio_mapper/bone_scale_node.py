"""
SDPose bone-scale node.

Applies per-bone length scaling to a POSE_KEYPOINT stream using a
forward-kinematics chain anchored at the neck.  Supports anisotropic
head/face X/Y scaling, bilateral shoulder/hip width, and extremity
(hand/foot) propagation.
"""

import copy
import numpy as np


# Keypoint body indices
_NOSE = 0
_NECK = 1
_R_SHOULDER = 2
_R_ELBOW = 3
_R_WRIST = 4
_L_SHOULDER = 5
_L_ELBOW = 6
_L_WRIST = 7
_R_HIP = 8
_R_KNEE = 9
_R_ANKLE = 10
_L_HIP = 11
_L_KNEE = 12
_L_ANKLE = 13
_L_EYE = 14
_R_EYE = 15
_L_EAR = 16
_R_EAR = 17

_HEAD_INDICES = [_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR]
_ALL_UPPER = [_R_SHOULDER, _R_ELBOW, _R_WRIST, _L_SHOULDER, _L_ELBOW, _L_WRIST]
_ALL_LOWER = [_R_HIP, _R_KNEE, _R_ANKLE, _L_HIP, _L_KNEE, _L_ANKLE]
_ALL_BODY = _ALL_UPPER + _ALL_LOWER

_FACE_CENTER_IDX = 30  # nose-tip landmark


class BodyRatioMapperSDPoseBoneScale:
    """Scale individual bones of an SDPose POSE_KEYPOINT person."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT", {"tooltip": "Input SDPose POSE_KEYPOINT"}),
            },
            "optional": {
                "person_idx": ("INT", {"default": -1, "min": -1, "max": 100,
                    "tooltip": "Person index to scale. -1 = all persons."}),
                "scale_head_x": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01,
                    "tooltip": "Head + face X scale (nose/eyes/ears/face relative to neck)"}),
                "scale_head_y": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01,
                    "tooltip": "Head + face Y scale (nose/eyes/ears/face relative to neck)"}),
                "scale_neck": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01,
                    "tooltip": "Neck length scale (shoulder_mid to neck distance)"}),
                "scale_shoulder_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01,
                    "tooltip": "Shoulder width (distance between L/R shoulders)"}),
                "scale_upper_arm": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01,
                    "tooltip": "Upper arm scale (shoulder to elbow, L+R linked)"}),
                "scale_lower_arm": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01,
                    "tooltip": "Lower arm scale (elbow to wrist, L+R linked)"}),
                "scale_torso": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01,
                    "tooltip": "Torso length scale (shoulder_mid to hip_mid)"}),
                "scale_hip_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01,
                    "tooltip": "Hip width (distance between L/R hips)"}),
                "scale_upper_leg": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01,
                    "tooltip": "Upper leg scale (hip to knee, L+R linked)"}),
                "scale_lower_leg": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01,
                    "tooltip": "Lower leg scale (knee to ankle, L+R linked)"}),
                "scale_foot": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01,
                    "tooltip": "Foot internal scale (3 foot points relative to ankle)"}),
                "scale_hand": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01,
                    "tooltip": "Hand internal scale (21 hand points relative to wrist, L+R linked)"}),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoint",)
    FUNCTION = "process"
    CATEGORY = "BodyRatioMapper"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process(self, pose_keypoint, person_idx=-1,
                scale_head_x=1.0, scale_head_y=1.0,
                scale_neck=1.0, scale_shoulder_width=1.0,
                scale_upper_arm=1.0, scale_lower_arm=1.0,
                scale_torso=1.0, scale_hip_width=1.0,
                scale_upper_leg=1.0, scale_lower_leg=1.0,
                scale_foot=1.0, scale_hand=1.0):

        # Fast path: all scales == 1.0
        if all(s == 1.0 for s in [scale_head_x, scale_head_y, scale_neck,
                scale_shoulder_width, scale_upper_arm, scale_lower_arm,
                scale_torso, scale_hip_width, scale_upper_leg,
                scale_lower_leg, scale_foot, scale_hand]):
            return (copy.deepcopy(pose_keypoint),)

        result = copy.deepcopy(pose_keypoint)
        for frame in result:
            people = frame.get("people", [])
            targets = range(len(people)) if person_idx == -1 else [person_idx]
            for pi in targets:
                if 0 <= pi < len(people):
                    self._scale_person(
                        people[pi],
                        scale_head_x, scale_head_y,
                        scale_neck, scale_shoulder_width,
                        scale_upper_arm, scale_lower_arm,
                        scale_torso, scale_hip_width,
                        scale_upper_leg, scale_lower_leg,
                        scale_foot, scale_hand,
                    )
        return (result,)

    # ------------------------------------------------------------------
    # Per-person scaling
    # ------------------------------------------------------------------

    def _scale_person(self, person,
                      s_hx, s_hy, s_neck, s_sh_w,
                      s_ua, s_la, s_torso, s_hip_w,
                      s_ul, s_ll, s_foot, s_hand):

        pose = self._triplets(person.get("pose_keypoints_2d", []), 18)
        face = self._triplets(person.get("face_keypoints_2d", []), 68)
        hand_l = person.get("hand_left_keypoints_2d", [])
        hand_r = person.get("hand_right_keypoints_2d", [])
        foot = person.get("foot_keypoints_2d", [])

        # Snapshot original XY for vector computation
        orig = [(pt[0], pt[1]) for pt in pose]
        face_orig = [(pt[0], pt[1]) for pt in face]

        neck_x, neck_y = orig[_NECK]

        # ---- Step 1: HEAD + FACE (X/Y anisotropic, anchored at neck) ----
        for idx in _HEAD_INDICES:
            if self._is_valid(pose[idx]):
                pose[idx][0] = neck_x + (orig[idx][0] - neck_x) * s_hx
                pose[idx][1] = neck_y + (orig[idx][1] - neck_y) * s_hy

        # Face: translate with nose delta, then scale around landmark 30
        nose_delta_x = pose[_NOSE][0] - orig[_NOSE][0]
        nose_delta_y = pose[_NOSE][1] - orig[_NOSE][1]
        if self._is_valid(face[_FACE_CENTER_IDX]):
            for i in range(68):
                if self._is_valid(face[i]):
                    face[i][0] += nose_delta_x
                    face[i][1] += nose_delta_y
            cx, cy = face[_FACE_CENTER_IDX][0], face[_FACE_CENTER_IDX][1]
            for i in range(68):
                if self._is_valid(face[i]):
                    face[i][0] = cx + (face[i][0] - cx) * s_hx
                    face[i][1] = cy + (face[i][1] - cy) * s_hy

        # ---- Step 2: NECK (neck→nose distance scale; neck stays, head+face follow) ----
        dx = (orig[_NOSE][0] - neck_x) * (s_neck - 1.0)
        dy = (orig[_NOSE][1] - neck_y) * (s_neck - 1.0)

        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            # Head points follow
            for idx in _HEAD_INDICES:
                if self._is_valid(pose[idx]):
                    pose[idx][0] += dx
                    pose[idx][1] += dy
            # Face follows
            for i in range(68):
                if self._is_valid(face[i]):
                    face[i][0] += dx
                    face[i][1] += dy
            # Neck, body, hands, feet stay in place

        # ---- Step 3: SHOULDER WIDTH ----
        if self._is_valid(pose[_R_SHOULDER]) and self._is_valid(pose[_L_SHOULDER]):
            mid_x = (pose[_R_SHOULDER][0] + pose[_L_SHOULDER][0]) * 0.5
            # Right side
            rdx = (pose[_R_SHOULDER][0] - mid_x) * (s_sh_w - 1.0)
            self._translate_indices(pose, [_R_SHOULDER, _R_ELBOW, _R_WRIST], rdx, 0.0)
            self._translate_flat(hand_r, 21, rdx, 0.0)
            # Left side
            ldx = (pose[_L_SHOULDER][0] - mid_x) * (s_sh_w - 1.0)
            self._translate_indices(pose, [_L_SHOULDER, _L_ELBOW, _L_WRIST], ldx, 0.0)
            self._translate_flat(hand_l, 21, ldx, 0.0)

        # ---- Step 4: UPPER ARM ----
        self._scale_chain(pose, orig, _R_SHOULDER, _R_ELBOW, s_ua, [_R_ELBOW, _R_WRIST], hand_r, 21)
        self._scale_chain(pose, orig, _L_SHOULDER, _L_ELBOW, s_ua, [_L_ELBOW, _L_WRIST], hand_l, 21)

        # ---- Step 5: LOWER ARM ----
        self._scale_chain(pose, orig, _R_ELBOW, _R_WRIST, s_la, [_R_WRIST], hand_r, 21)
        self._scale_chain(pose, orig, _L_ELBOW, _L_WRIST, s_la, [_L_WRIST], hand_l, 21)

        # ---- Step 6: TORSO ----
        # Use ORIGINAL shoulder_mid and hip_mid for direction vector
        if (self._is_valid(pose[_R_SHOULDER]) and self._is_valid(pose[_L_SHOULDER])
                and self._is_valid(pose[_R_HIP]) and self._is_valid(pose[_L_HIP])):
            sm_orig_x = (orig[_R_SHOULDER][0] + orig[_L_SHOULDER][0]) * 0.5
            sm_orig_y = (orig[_R_SHOULDER][1] + orig[_L_SHOULDER][1]) * 0.5
            hm_orig_x = (orig[_R_HIP][0] + orig[_L_HIP][0]) * 0.5
            hm_orig_y = (orig[_R_HIP][1] + orig[_L_HIP][1]) * 0.5
            # Current shoulder_mid (may have been moved by neck step)
            sm_cur_x = (pose[_R_SHOULDER][0] + pose[_L_SHOULDER][0]) * 0.5
            sm_cur_y = (pose[_R_SHOULDER][1] + pose[_L_SHOULDER][1]) * 0.5
            vec_x = hm_orig_x - sm_orig_x
            vec_y = hm_orig_y - sm_orig_y
            new_hm_x = sm_cur_x + vec_x * s_torso
            new_hm_y = sm_cur_y + vec_y * s_torso
            dx = new_hm_x - hm_orig_x
            dy = new_hm_y - hm_orig_y
            self._translate_indices(pose, _ALL_LOWER, dx, dy)
            self._translate_flat(foot, 6, dx, dy)

        # ---- Step 7: HIP WIDTH ----
        if self._is_valid(pose[_R_HIP]) and self._is_valid(pose[_L_HIP]):
            mid_x = (pose[_R_HIP][0] + pose[_L_HIP][0]) * 0.5
            # Right side
            rdx = (pose[_R_HIP][0] - mid_x) * (s_hip_w - 1.0)
            self._translate_indices(pose, [_R_HIP, _R_KNEE, _R_ANKLE], rdx, 0.0)
            self._translate_foot_side(foot, 'right', rdx, 0.0)
            # Left side
            ldx = (pose[_L_HIP][0] - mid_x) * (s_hip_w - 1.0)
            self._translate_indices(pose, [_L_HIP, _L_KNEE, _L_ANKLE], ldx, 0.0)
            self._translate_foot_side(foot, 'left', ldx, 0.0)

        # ---- Step 8: UPPER LEG ----
        self._scale_chain(pose, orig, _R_HIP, _R_KNEE, s_ul, [_R_KNEE, _R_ANKLE], foot, 6, foot_side='right')
        self._scale_chain(pose, orig, _L_HIP, _L_KNEE, s_ul, [_L_KNEE, _L_ANKLE], foot, 6, foot_side='left')

        # ---- Step 9: LOWER LEG ----
        self._scale_chain(pose, orig, _R_KNEE, _R_ANKLE, s_ll, [_R_ANKLE], foot, 6, foot_side='right')
        self._scale_chain(pose, orig, _L_KNEE, _L_ANKLE, s_ll, [_L_ANKLE], foot, 6, foot_side='left')

        # ---- Step 10: FOOT (internal, anchored at ankle) ----
        self._scale_foot_points(pose, foot, _R_ANKLE, 3, 5, s_foot)  # right foot: indices 3,4,5
        self._scale_foot_points(pose, foot, _L_ANKLE, 0, 2, s_foot)  # left foot:  indices 0,1,2

        # ---- Step 11: HAND (internal, anchored at wrist) ----
        self._scale_hand_points(hand_r, pose, _R_WRIST, s_hand)  # right hand: 21 points
        self._scale_hand_points(hand_l, pose, _L_WRIST, s_hand)  # left hand:  21 points

        # ---- Writeback ----
        self._writeback(person.get("pose_keypoints_2d", []), pose, 18)
        self._writeback(person.get("face_keypoints_2d", []), face, 68)

    # ------------------------------------------------------------------
    # Chain scaling helper
    # ------------------------------------------------------------------

    def _scale_chain(self, pose, orig, parent_idx, child_idx, scale,
                     propagate_indices, flat_ext, ext_count,
                     foot_side=None):
        """Scale parent→child vector, propagate delta to children and extremity."""
        if not (self._is_valid(pose[parent_idx]) and self._is_valid(pose[child_idx])):
            return
        vec_x = (orig[child_idx][0] - orig[parent_idx][0]) * scale
        vec_y = (orig[child_idx][1] - orig[parent_idx][1]) * scale
        new_x = pose[parent_idx][0] + vec_x
        new_y = pose[parent_idx][1] + vec_y
        dx = new_x - pose[child_idx][0]
        dy = new_y - pose[child_idx][1]
        for idx in propagate_indices:
            if self._is_valid(pose[idx]):
                pose[idx][0] += dx
                pose[idx][1] += dy
        if flat_ext is not None:
            if foot_side is not None:
                self._translate_foot_side(flat_ext, foot_side, dx, dy)
            else:
                self._translate_flat(flat_ext, ext_count, dx, dy)

    # ------------------------------------------------------------------
    # Foot scaling helper
    # ------------------------------------------------------------------

    def _scale_foot_points(self, pose, foot_flat, ankle_idx, foot_start, foot_end, scale):
        """Scale foot points [foot_start..foot_end] around ankle."""
        if not self._is_valid(pose[ankle_idx]):
            return
        ax, ay = pose[ankle_idx][0], pose[ankle_idx][1]
        for i in range(foot_start, foot_end + 1):
            idx = i * 3
            if idx + 1 >= len(foot_flat):
                continue
            if abs(foot_flat[idx]) > 1e-3 or abs(foot_flat[idx + 1]) > 1e-3:
                foot_flat[idx] = ax + (foot_flat[idx] - ax) * scale
                foot_flat[idx + 1] = ay + (foot_flat[idx + 1] - ay) * scale

    def _scale_hand_points(self, hand_flat, pose, wrist_idx, scale):
        """Scale 21 hand points around wrist."""
        if not isinstance(hand_flat, list) or not self._is_valid(pose[wrist_idx]):
            return
        wx, wy = pose[wrist_idx][0], pose[wrist_idx][1]
        for i in range(21):
            idx = i * 3
            if idx + 1 >= len(hand_flat):
                continue
            if abs(hand_flat[idx]) > 1e-3 or abs(hand_flat[idx + 1]) > 1e-3:
                hand_flat[idx] = wx + (hand_flat[idx] - wx) * scale
                hand_flat[idx + 1] = wy + (hand_flat[idx + 1] - wy) * scale

    # ------------------------------------------------------------------
    # Translation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _translate_indices(pose, indices, dx, dy):
        for idx in indices:
            if abs(pose[idx][0]) > 1e-3 or abs(pose[idx][1]) > 1e-3:
                pose[idx][0] += dx
                pose[idx][1] += dy

    @staticmethod
    def _translate_flat(flat_arr, count, dx, dy):
        if not isinstance(flat_arr, list):
            return
        for i in range(count):
            idx = i * 3
            if idx + 1 >= len(flat_arr):
                continue
            if abs(flat_arr[idx]) > 1e-3 or abs(flat_arr[idx + 1]) > 1e-3:
                flat_arr[idx] += dx
                flat_arr[idx + 1] += dy

    @staticmethod
    def _translate_foot_side(foot_flat, side, dx, dy):
        """Translate left (indices 0-2) or right (indices 3-5) foot points."""
        if not isinstance(foot_flat, list):
            return
        start = 0 if side == 'left' else 3
        end = 3 if side == 'left' else 6
        for i in range(start, end):
            idx = i * 3
            if idx + 1 >= len(foot_flat):
                continue
            if abs(foot_flat[idx]) > 1e-3 or abs(foot_flat[idx + 1]) > 1e-3:
                foot_flat[idx] += dx
                foot_flat[idx + 1] += dy

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _triplets(arr, count):
        if not isinstance(arr, list):
            arr = []
        need = count * 3
        if len(arr) < need:
            arr = arr + [0.0] * (need - len(arr))
        return [[arr[i * 3], arr[i * 3 + 1], arr[i * 3 + 2]] for i in range(count)]

    @staticmethod
    def _is_valid(pt):
        return abs(pt[0]) > 1e-3 or abs(pt[1]) > 1e-3

    @staticmethod
    def _writeback(flat_arr, triplets, count):
        if not isinstance(flat_arr, list):
            return
        for i in range(count):
            idx = i * 3
            if idx + 1 >= len(flat_arr):
                continue
            flat_arr[idx] = float(triplets[i][0])
            flat_arr[idx + 1] = float(triplets[i][1])


class BodyRatioMapperSDPoseTranslate:
    """Translate all keypoints of an SDPose POSE_KEYPOINT person."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT", {"tooltip": "Input SDPose POSE_KEYPOINT"}),
            },
            "optional": {
                "person_idx": ("INT", {"default": -1, "min": -1, "max": 100,
                    "tooltip": "Person index to translate. -1 = all persons."}),
                "translate_x": ("FLOAT", {"default": 0.0, "min": -2000.0, "max": 2000.0, "step": 1.0,
                    "tooltip": "X offset in pixels"}),
                "translate_y": ("FLOAT", {"default": 0.0, "min": -2000.0, "max": 2000.0, "step": 1.0,
                    "tooltip": "Y offset in pixels"}),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoint",)
    FUNCTION = "process"
    CATEGORY = "BodyRatioMapper"

    def process(self, pose_keypoint, person_idx=-1, translate_x=0.0, translate_y=0.0):
        if translate_x == 0.0 and translate_y == 0.0:
            return (copy.deepcopy(pose_keypoint),)

        result = copy.deepcopy(pose_keypoint)
        dx, dy = float(translate_x), float(translate_y)
        for frame in result:
            people = frame.get("people", [])
            targets = range(len(people)) if person_idx == -1 else [person_idx]
            for pi in targets:
                if 0 <= pi < len(people):
                    self._translate_person(people[pi], dx, dy)
        return (result,)

    @staticmethod
    def _translate_person(person, dx, dy):
        for key in ("pose_keypoints_2d", "face_keypoints_2d",
                     "hand_left_keypoints_2d", "hand_right_keypoints_2d",
                     "foot_keypoints_2d"):
            arr = person.get(key)
            if not isinstance(arr, list):
                continue
            for i in range(0, len(arr) - 1, 3):
                if abs(arr[i]) > 1e-3 or abs(arr[i + 1]) > 1e-3:
                    arr[i] += dx
                    arr[i + 1] += dy
