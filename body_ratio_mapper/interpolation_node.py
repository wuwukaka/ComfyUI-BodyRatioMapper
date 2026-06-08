# Copyright (C) 2026 wuwukasi (wuwukaka)
# SPDX-License-Identifier: GPL-3.0-only

"""
SDPose interpolation node.

Linearly interpolates between the last frame of a first POSE_KEYPOINT
stream and the first frame of a second stream, producing a smooth
transition with a configurable number of intermediate frames.
"""

import copy

# Keys in a person dict that hold flat keypoint arrays (stride-3 triplets).
_KP_KEYS = [
    "pose_keypoints_2d",
    "face_keypoints_2d",
    "hand_left_keypoints_2d",
    "hand_right_keypoints_2d",
    "foot_keypoints_2d",
]


class BodyRatioMapperSDPoseInterpolate:
    """Linear interpolation between two POSE_KEYPOINT streams."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint_a": ("POSE_KEYPOINT", {"tooltip": "First keypoint group (Group A)"}),
                "pose_keypoint_b": ("POSE_KEYPOINT", {"tooltip": "Second keypoint group (Group B)"}),
            },
            "optional": {
                "frame_count": ("INT", {
                    "default": 5, "min": 1, "max": 120,
                    "tooltip": "Number of interpolated frames between the last frame of A and first frame of B",
                }),
                "person_idx": ("INT", {
                    "default": -1, "min": -1, "max": 100,
                    "tooltip": "Person index to interpolate. -1 = all persons.",
                }),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoint",)
    FUNCTION = "process"
    CATEGORY = "BodyRatioMapper"

    def process(self, pose_keypoint_a, pose_keypoint_b,
                frame_count=5, person_idx=-1):
        # --- 1. Deep-copy inputs for immutability ---
        a = copy.deepcopy(pose_keypoint_a)
        b = copy.deepcopy(pose_keypoint_b)

        # --- 2. Edge cases: one or both empty ---
        if not a and not b:
            return ([],)
        if not a:
            return (b,)
        if not b:
            return (a,)

        # --- 3. Extract anchor frames ---
        frame_a = a[-1]
        frame_b = b[0]

        people_a = frame_a.get("people", [])
        people_b = frame_b.get("people", [])

        # --- 4. Edge case: no people on either side ---
        if not people_a and not people_b:
            return (a + b,)
        if not people_a or not people_b:
            return (a + b,)

        # --- 5. Canvas size for interpolated frames ---
        canvas_width = frame_a.get("canvas_width", 512)
        canvas_height = frame_a.get("canvas_height", 768)

        # --- 6. Determine which person indices to interpolate ---
        n_a = len(people_a)
        n_b = len(people_b)

        if person_idx == -1:
            interp_set = set(range(min(n_a, n_b)))
        else:
            interp_set = {person_idx} if person_idx < min(n_a, n_b) else set()

        # --- 7. Generate interpolated frames ---
        n_total = max(n_a, n_b)
        interp_frames = []

        for i in range(1, frame_count + 1):
            w = i / (frame_count + 1)

            interp_frame = {
                "canvas_width": canvas_width,
                "canvas_height": canvas_height,
                "people": [],
            }

            for p_idx in range(n_total):
                if p_idx in interp_set:
                    interp_person = self._interpolate_person(
                        people_a[p_idx], people_b[p_idx], w)
                elif p_idx < n_a:
                    interp_person = copy.deepcopy(people_a[p_idx])
                else:
                    interp_person = copy.deepcopy(people_b[p_idx])
                interp_frame["people"].append(interp_person)

            interp_frames.append(interp_frame)

        # --- 8. Assemble output: all of A + interpolated + all of B ---
        result = a + interp_frames + b
        return (result,)

    @staticmethod
    def _interpolate_person(person_a, person_b, w):
        """Interpolate all 5 keypoint arrays between two person dicts with weight w."""
        person = {}
        for key in _KP_KEYS:
            arr_a = person_a.get(key, [])
            arr_b = person_b.get(key, [])

            if not arr_a and not arr_b:
                person[key] = []
                continue

            max_len = max(len(arr_a), len(arr_b))
            padded_a = (arr_a + [0.0] * (max_len - len(arr_a))
                        if len(arr_a) < max_len else list(arr_a))
            padded_b = (arr_b + [0.0] * (max_len - len(arr_b))
                        if len(arr_b) < max_len else list(arr_b))

            inv_w = 1.0 - w
            interpolated = []
            for j in range(0, max_len, 3):
                x = padded_a[j] * inv_w + padded_b[j] * w
                y = padded_a[j + 1] * inv_w + padded_b[j + 1] * w
                c = min(padded_a[j + 2], padded_b[j + 2])
                interpolated.extend([x, y, c])
            person[key] = interpolated

        return person
