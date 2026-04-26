"""
BodyRatioMapper proportion transfer node.
Supports SDPose/OpenPose parsing, anchor selection, FK/RPCA scale solving,
time-offset stabilization, and batch pose transformation.
"""

import numpy as np
import copy
import io
import contextlib
from dataclasses import dataclass
from .core_modules.wscs_anchor import select_anchor as select_wscs_anchor
from .core_modules.global_rpca import (
    calculate_13_bone_global_rpca as calculate_global_rpca_external,
    forge_final_scale_constants as forge_final_scales_external,
)
from .core_modules import frame_ops as frame_ops_external
from .core_modules import scale_solver as scale_solver_external
from .core_modules import matrix_ops as matrix_ops_external


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime switches and thresholds for proportion processing."""
    alignment_mode: bool = False
    hand_scaling: bool = True
    foot_scaling: bool = True
    offset_stabilizer: bool = True
    offset_stabilizer_x: bool = False
    best_hand_search: bool = True
    use_shoulder_fk_for_hand: bool = False
    best_neck_search: bool = False
    final_offset_alignment: bool = True
    first_frame_offset_alignment: bool = False
    anchor_output_mode: str = "single_frame_multi_person"
    print_detailed_logs: bool = False
    confidence_threshold: float = 0.30
    output_absolute_coordinates: bool = True

class BodyRatioMapperProportionTransfer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "pose_keypoint": ("POSE_KEYPOINT", {"tooltip": "Target pose keypoints (SDPose or OpenPose format)"}),
            },
            "optional": {
                "ref_pose_keypoint": ("POSE_KEYPOINT", {"tooltip": "Reference pose keypoint (Source of proportions)"}),
                "manual_anchor_pose": ("POSE_KEYPOINT", {"tooltip": "(Optional) Override WSCS. Provide a perfect straight-facing pose to set the baseline proportions."}),
                "alignment_mode": ("BOOLEAN", {"default": False, "label_on": "Alignment Mode", "label_off": "Standard Mode"}),
                "hand_scaling": ("BOOLEAN", {"default": True, "label_on": "Hand Scaling ON", "label_off": "Hand Scaling OFF"}),
                "foot_scaling": ("BOOLEAN", {"default": True, "label_on": "Foot Scaling ON", "label_off": "Foot Scaling OFF"}),
                "offset_stabilizer": ("BOOLEAN", {"default": True, "label_on": "Offset Stabilizer ON", "label_off": "Offset Stabilizer OFF"}),
                "offset_stabilizer_x": ("BOOLEAN", {"default": False, "label_on": "Offset Stabilizer X ON", "label_off": "Offset Stabilizer X OFF"}),
                "best_hand_search": ("BOOLEAN", {"default": True, "label_on": "Best Hand Search ON", "label_off": "Best Hand Search OFF"}),
                "use_shoulder_fk_for_hand": ("BOOLEAN", {"default": False, "label_on": "Shoulder FK For Hand ON", "label_off": "Shoulder FK For Hand OFF", "tooltip": "Use shoulder FK to replace hand FK."}),
                "best_neck_search": ("BOOLEAN", {"default": False, "label_on": "Best Neck Search ON", "label_off": "Best Neck Search OFF"}),
                "final_offset_alignment": ("BOOLEAN", {"default": True, "label_on": "Final Offset Align ON", "label_off": "Final Offset Align OFF"}),
                "first_frame_offset_alignment": ("BOOLEAN", {"default": False, "label_on": "First-Frame Offset", "label_off": "Anchor-Frame Offset", "tooltip": "ON: align global offset by reference vs first frame. OFF: align by reference vs anchor frame."}),
                "anchor_output_mode": (["single_frame_multi_person", "multi_frame_single_person"], {"default": "single_frame_multi_person", "tooltip": "Anchor output layout mode."}),
                "print_detailed_logs": ("BOOLEAN", {"default": False, "label_on": "Detailed Logs ON", "label_off": "Detailed Logs OFF", "tooltip": "Enable verbose internal logs. OFF uses concise summary logs."}),
                "confidence_threshold": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Unified confidence threshold for WSCS and reference gating."}),
                "output_absolute_coordinates": ("BOOLEAN", {"default": True, "label_on": "Absolute Pixels", "label_off": "Normalized (0-1)"}),
            },
        }

    # Outputs transformed keypoints and selected anchor keypoints.
    RETURN_TYPES = ("POSE_KEYPOINT", "POSE_KEYPOINT")
    RETURN_NAMES = ("changed_pose_keypoint", "anchor_pose_keypoint")
    FUNCTION = "process"
    CATEGORY = "BodyRatioMapper"

    @staticmethod
    def _get_physical_dist(p1, p2):
        """Calculate Euclidean distance between two points in physical pixel coordinates"""
        if not isinstance(p1, np.ndarray):
            p1 = np.array(p1)
        if not isinstance(p2, np.ndarray):
            p2 = np.array(p2)
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    @staticmethod
    def _log_multi_summary(event, **fields):
        ordered = " ".join([f"{k}={fields[k]}" for k in sorted(fields.keys())])
        print(f"[MultiSummary] event={event} {ordered}".strip())

    @staticmethod
    def _build_empty_frame_container():
        """Build an empty internal frame container with fixed tensor shapes."""
        return {
            'bodies': {
                'candidate': np.zeros((18, 2)),
                'subset': [],
                'candidate_conf': np.zeros((18,))
            },
            'faces': np.zeros((1, 68, 2)),
            'faces_conf': np.zeros((1, 68)),
            'hands': np.zeros((2, 21, 2)),
            'hands_conf': np.zeros((2, 21)),
            'feet': np.zeros((2, 3, 2)),
            'feet_conf': np.zeros((2, 3))
        }

    @staticmethod
    def _build_runtime_config(alignment_mode=False,
                              hand_scaling=True,
                              foot_scaling=True,
                              offset_stabilizer=True,
                              offset_stabilizer_x=False,
                              best_hand_search=True,
                              use_shoulder_fk_for_hand=False,
                              best_neck_search=False,
                              final_offset_alignment=True,
                              first_frame_offset_alignment=False,
                              anchor_output_mode="single_frame_multi_person",
                              print_detailed_logs=False,
                              confidence_threshold=0.30,
                              output_absolute_coordinates=True):
        """Build RuntimeConfig from node input values."""
        return RuntimeConfig(
            alignment_mode=bool(alignment_mode),
            hand_scaling=bool(hand_scaling),
            foot_scaling=bool(foot_scaling),
            offset_stabilizer=bool(offset_stabilizer),
            offset_stabilizer_x=bool(offset_stabilizer_x),
            best_hand_search=bool(best_hand_search),
            use_shoulder_fk_for_hand=bool(use_shoulder_fk_for_hand),
            best_neck_search=bool(best_neck_search),
            final_offset_alignment=bool(final_offset_alignment),
            first_frame_offset_alignment=bool(first_frame_offset_alignment),
            anchor_output_mode=str(anchor_output_mode),
            print_detailed_logs=bool(print_detailed_logs),
            confidence_threshold=float(confidence_threshold),
            output_absolute_coordinates=bool(output_absolute_coordinates),
        )

    @staticmethod
    def _resolve_runtime_config(config=None,
                                alignment_mode=False,
                                hand_scaling=True,
                                foot_scaling=True,
                                offset_stabilizer=True,
                                offset_stabilizer_x=False,
                                best_hand_search=True,
                                use_shoulder_fk_for_hand=False,
                                best_neck_search=False,
                                final_offset_alignment=True,
                                first_frame_offset_alignment=False,
                                anchor_output_mode="single_frame_multi_person",
                                print_detailed_logs=False,
                                confidence_threshold=0.30,
                                output_absolute_coordinates=True):
        """Resolve runtime config from explicit object or parameter set."""
        if config is not None:
            return config
        return BodyRatioMapperProportionTransfer._build_runtime_config(
            alignment_mode=alignment_mode,
            hand_scaling=hand_scaling,
            foot_scaling=foot_scaling,
            offset_stabilizer=offset_stabilizer,
            offset_stabilizer_x=offset_stabilizer_x,
            best_hand_search=best_hand_search,
            use_shoulder_fk_for_hand=use_shoulder_fk_for_hand,
            best_neck_search=best_neck_search,
            final_offset_alignment=final_offset_alignment,
            first_frame_offset_alignment=first_frame_offset_alignment,
            anchor_output_mode=anchor_output_mode,
            print_detailed_logs=print_detailed_logs,
            confidence_threshold=confidence_threshold,
            output_absolute_coordinates=output_absolute_coordinates,
        )

    def _parse_single_person(self, person, canvas_w, canvas_h):
        """Parse one OpenPose/SDPose person object into internal numpy tensors."""
        frame_container = self._build_empty_frame_container()

        # --- Body Parsing (OpenPose 18) ---
        pose_2d = person.get('pose_keypoints_2d', [])

        # Prepare variable to store wrist confidence for hand processing later
        # Default to 0.0 if not found
        wrist_l_conf_val = 0.0  # Index 7
        wrist_r_conf_val = 0.0  # Index 4

        if len(pose_2d) > 0:
            kpts_all = np.array(pose_2d).reshape(-1, 3)
            body_xy = kpts_all[:, :2]
            body_conf = kpts_all[:, 2]  # Extract confidence

            # Convert to physical coordinates if normalized
            if np.max(body_xy) <= 1.5:
                body_xy[:, 0] *= canvas_w
                body_xy[:, 1] *= canvas_h

            # Parser confidence gate for body points.
            # Note: We clear XY but keep the conf array aligned for reference
            body_xy[body_conf < 0.05] = 0.0

            # Extract Wrist Confidences for Hand Logic (Indices 4 and 7)
            if len(body_conf) > 7:
                wrist_l_conf_val = body_conf[7]
            if len(body_conf) > 4:
                wrist_r_conf_val = body_conf[4]

            if body_xy.shape[0] < 18:
                pad_xy = np.zeros((18 - body_xy.shape[0], 2))
                pad_conf = np.zeros((18 - body_xy.shape[0],))
                body_xy = np.vstack((body_xy, pad_xy))
                body_conf = np.concatenate((body_conf, pad_conf))

            frame_container['bodies']['candidate'] = body_xy[:18]
            frame_container['bodies']['candidate_conf'] = body_conf[:18]

        # --- Feet Parsing (Indices 18-23 or foot_keypoints) ---
        feet_arr = np.zeros((2, 3, 2))  # [Left, Right]
        feet_conf_arr = np.zeros((2, 3))
        foot_2d = person.get('foot_keypoints_2d', [])

        # Scenario A: Standard OpenPose
        if len(foot_2d) > 0:
            kpts_all = np.array(foot_2d).reshape(-1, 3)
            kpts = kpts_all[:, :2]
            conf = kpts_all[:, 2]

            # Convert to physical coordinates if normalized
            if np.max(kpts) <= 1.5:
                kpts[:, 0] *= canvas_w
                kpts[:, 1] *= canvas_h

            # Parser confidence gate for foot points.
            kpts[conf < 0.05] = 0.0

            if len(kpts) >= 6:
                feet_arr[0] = kpts[:3]  # Left
                feet_conf_arr[0] = conf[:3]
                feet_arr[1] = kpts[3:6]  # Right
                feet_conf_arr[1] = conf[3:6]
        # Scenario B: SDPose WholeBody (Indices 18-23)
        elif len(pose_2d) >= (24 * 3):
            kpts_all = np.array(pose_2d).reshape(-1, 3)
            kpts_xy = kpts_all[:, :2]
            kpts_conf = kpts_all[:, 2]

            # Convert to physical coordinates if normalized
            if np.max(kpts_xy) <= 1.5:
                kpts_xy[:, 0] *= canvas_w
                kpts_xy[:, 1] *= canvas_h

            # Parser confidence gate for face points.
            kpts_xy[kpts_conf < 0.05] = 0.0

            feet_arr[0] = kpts_xy[18:21]  # Left
            feet_conf_arr[0] = kpts_conf[18:21]
            feet_arr[1] = kpts_xy[21:24]  # Right
            feet_conf_arr[1] = kpts_conf[21:24]

        frame_container['feet'] = feet_arr
        frame_container['feet_conf'] = feet_conf_arr

        # --- Face Parsing ---
        face_2d = person.get('face_keypoints_2d', [])
        if len(face_2d) > 0:
            kpts_all = np.array(face_2d).reshape(-1, 3)
            face_xy = kpts_all[:, :2]
            face_conf = kpts_all[:, 2]

            # Convert to physical coordinates if normalized
            if np.max(face_xy) <= 1.5:
                face_xy[:, 0] *= canvas_w
                face_xy[:, 1] *= canvas_h

            # Parser confidence gate for hand points.
            face_xy[face_conf < 0.05] = 0.0
            frame_container['faces'] = face_xy[np.newaxis, :, :]
            frame_container['faces_conf'] = face_conf[np.newaxis, :]

        # --- Hands Parsing (2x21 points) ---
        # Removed hard dependency check (if wrist_ok).
        # Soft dependency: hand confidence is capped by wrist confidence.
        # Single-Head Error Correction: Cross-extract hands to bypass upstream bug.
        hands_arr = np.zeros((2, 21, 2))
        hands_conf_arr = np.zeros((2, 21))

        # 1. Processing LEFT Hand Input (Upstream Bug: Real Left is inside 'hand_right_keypoints_2d')
        # LOGIC: Input Left Hand maps to Index 0 (Physically linked to LEFT Wrist, Body Index 7)
        lh_2d = person.get('hand_right_keypoints_2d', [])
        if len(lh_2d) > 0:
            kpts_all = np.array(lh_2d).reshape(-1, 3)
            kpts = kpts_all[:, :2]
            conf = kpts_all[:, 2]

            # Convert to physical coordinates if normalized
            if np.max(kpts) <= 1.5:
                kpts[:, 0] *= canvas_w
                kpts[:, 1] *= canvas_h

            # Parser confidence gate for left-hand points.
            kpts[conf < 0.05] = 0.0

            # Soft Dependency Cap: Cap hand confidence by Left Wrist (Index 7) confidence
            conf = np.minimum(conf, wrist_l_conf_val)

            if kpts.shape[0] >= 21:
                hands_arr[0] = kpts[:21]  # Index 0
                hands_conf_arr[0] = conf[:21]  # Stored with capped confidence

        # 2. Processing RIGHT Hand Input (Upstream Bug: Real Right is inside 'hand_left_keypoints_2d')
        # LOGIC: Input Right Hand maps to Index 1 (Physically linked to RIGHT Wrist, Body Index 4)
        rh_2d = person.get('hand_left_keypoints_2d', [])
        if len(rh_2d) > 0:
            kpts_all = np.array(rh_2d).reshape(-1, 3)
            kpts = kpts_all[:, :2]
            conf = kpts_all[:, 2]

            # Convert to physical coordinates if normalized
            if np.max(kpts) <= 1.5:
                kpts[:, 0] *= canvas_w
                kpts[:, 1] *= canvas_h

            # Parser confidence gate for right-hand points.
            kpts[conf < 0.05] = 0.0

            # Soft Dependency Cap: Cap hand confidence by Right Wrist (Index 4) confidence
            conf = np.minimum(conf, wrist_r_conf_val)

            if kpts.shape[0] >= 21:
                hands_arr[1] = kpts[:21]  # Index 1
                hands_conf_arr[1] = conf[:21]  # Stored with capped confidence

        frame_container['hands'] = hands_arr
        frame_container['hands_conf'] = hands_conf_arr
        return frame_container

    def parse_keypoints(self, pose_data, width=512, height=768):
        """
        Parses POSE_KEYPOINT data (dictionary list) into the NumPy format required by the algorithm.
        Supports both SDPose (Normalized) and OpenPose (Absolute) formats.
        Extracts and stores confidence values separately.
        Confidence arrays are preserved and hand confidence is wrist-capped.
        """
        parsed_batch = []
        
        if not pose_data:
            return parsed_batch

        for frame in pose_data:
            canvas_w = frame.get('canvas_width', width)
            canvas_h = frame.get('canvas_height', height)
            people = frame.get('people', [])

            if not people:
                parsed_batch.append(self._build_empty_frame_container())
                continue

            person = people[0]
            parsed_batch.append(self._parse_single_person(person, canvas_w, canvas_h))

        return parsed_batch

    def _serialize_single_frame(self, processed, orig_frame):
        """Serialize one processed frame back to SDPose/OpenPose JSON frame."""
        canvas_w = orig_frame.get('canvas_width', 512)
        canvas_h = orig_frame.get('canvas_height', 768)

        # Helper: STRICT CLEANING for 0.0 points
        def get_val_with_conf(pt, conf_val):
            # If point is effectively zero, return pure zero coordinates and zero confidence
            if abs(pt[0]) < 1e-3 and abs(pt[1]) < 1e-3:
                return [0.0, 0.0, 0.0]
            # Return [x, y, original_confidence]
            return [float(pt[0]), float(pt[1]), float(conf_val)]

        # Body
        body_np = processed['bodies']['candidate']
        body_conf = processed['bodies']['candidate_conf']
        body_flat = []
        for j, point in enumerate(body_np):
            c = body_conf[j] if j < len(body_conf) else 0.0
            body_flat.extend(get_val_with_conf(point, c))

        # Feet
        left_foot = processed['feet'][0]
        left_conf = processed['feet_conf'][0]
        right_foot = processed['feet'][1]
        right_conf = processed['feet_conf'][1]

        feet_flat = []
        for j, p in enumerate(left_foot):
            c = left_conf[j] if j < len(left_conf) else 0.0
            feet_flat.extend(get_val_with_conf(p, c))
        for j, p in enumerate(right_foot):
            c = right_conf[j] if j < len(right_conf) else 0.0
            feet_flat.extend(get_val_with_conf(p, c))

        # Face
        face_flat = []
        if processed['faces'].shape[1] > 0:
            face_np = processed['faces'][0]
            face_conf = processed['faces_conf'][0]
            for j, p in enumerate(face_np):
                c = face_conf[j] if j < len(face_conf) else 0.0
                face_flat.extend(get_val_with_conf(p, c))

        # Hands
        hands_np = processed['hands']
        hands_conf = processed['hands_conf']
        left_hand_flat = []
        right_hand_flat = []

        # Index 0 (Left Hand data, moved by Left Arm logic) -> Write to Left Key
        for j, p in enumerate(hands_np[0]):
            c = hands_conf[0][j] if j < len(hands_conf[0]) else 0.0
            left_hand_flat.extend(get_val_with_conf(p, c))

        # Index 1 (Right Hand data, moved by Right Arm logic) -> Write to Right Key
        for j, p in enumerate(hands_np[1]):
            c = hands_conf[1][j] if j < len(hands_conf[1]) else 0.0
            right_hand_flat.extend(get_val_with_conf(p, c))

        person = {
            "pose_keypoints_2d": body_flat,
            "face_keypoints_2d": face_flat,
            "hand_left_keypoints_2d": left_hand_flat,
            "hand_right_keypoints_2d": right_hand_flat,
            "foot_keypoints_2d": feet_flat
        }
        return {
            "people": [person],
            "canvas_width": canvas_w,
            "canvas_height": canvas_h
        }

    def serialize_to_sdpose(self, processed_batch, original_frames):
        """
        Converts the processed NumPy data back into SDPose/OpenPose JSON format.
        Uses stored confidence values.
        """
        result_frames = []
        
        for i, processed in enumerate(processed_batch):
            orig_frame = original_frames[i] if i < len(original_frames) else original_frames[0]
            result_frames.append(self._serialize_single_frame(processed, orig_frame))
            
        return result_frames

    @staticmethod
    def _assert_single_frame_input(name, value):
        if value is None:
            return
        if (not isinstance(value, list)) or len(value) != 1:
            raise ValueError(f"{name} must contain exactly one frame.")

    @staticmethod
    def _build_zero_person_openpose():
        def z(n):
            return [0.0] * (n * 3)
        return {
            "pose_keypoints_2d": z(18),
            "face_keypoints_2d": z(70),
            "hand_left_keypoints_2d": z(21),
            "hand_right_keypoints_2d": z(21),
            "foot_keypoints_2d": z(6),
        }

    def _clone_person_fast(self, person):
        """
        Fast clone for person dict with fixed flat keypoint arrays.
        Keeps behavior-equivalent payload isolation without deepcopy overhead.
        """
        src = self._normalize_person_schema(person)
        out = dict(src)
        out["pose_keypoints_2d"] = list(src.get("pose_keypoints_2d", []))
        out["face_keypoints_2d"] = list(src.get("face_keypoints_2d", []))
        out["hand_left_keypoints_2d"] = list(src.get("hand_left_keypoints_2d", []))
        out["hand_right_keypoints_2d"] = list(src.get("hand_right_keypoints_2d", []))
        out["foot_keypoints_2d"] = list(src.get("foot_keypoints_2d", []))
        return out

    def _clone_track_fast(self, track):
        """Fast clone for single-person track frames (people[0] + canvas metadata)."""
        if not isinstance(track, list):
            return []
        out = []
        for frame in track:
            people = frame.get("people", []) if isinstance(frame, dict) else []
            person = people[0] if isinstance(people, list) and len(people) > 0 else self._build_zero_person_openpose()
            out.append({
                "people": [self._clone_person_fast(person)],
                "canvas_width": frame.get("canvas_width", 512) if isinstance(frame, dict) else 512,
                "canvas_height": frame.get("canvas_height", 768) if isinstance(frame, dict) else 768,
            })
        return out

    def _normalize_person_schema(self, person):
        if not isinstance(person, dict):
            return self._build_zero_person_openpose()
        norm = dict(person)
        defaults = self._build_zero_person_openpose()
        for key, val in defaults.items():
            arr = norm.get(key, val)
            if not isinstance(arr, list):
                arr = list(val)
            # Enforce fixed flat lengths expected by downstream parsers.
            target_len = len(val)
            if len(arr) < target_len:
                arr = arr + [0.0] * (target_len - len(arr))
            elif len(arr) > target_len:
                arr = arr[:target_len]
            norm[key] = arr
        return norm

    @staticmethod
    def _triplets(arr, count):
        if not isinstance(arr, list):
            arr = []
        need = count * 3
        if len(arr) < need:
            arr = arr + [0.0] * (need - len(arr))
        return [arr[i * 3:(i + 1) * 3] for i in range(count)]

    def _body18_triplets(self, person):
        person = self._normalize_person_schema(person)
        return self._triplets(person.get("pose_keypoints_2d", []), 18)

    def _count_body18_low_conf(self, person, conf_thresh):
        pose = self._body18_triplets(person)
        low = 0
        for i in range(18):
            c = float(pose[i][2]) if len(pose[i]) > 2 else 0.0
            if c < conf_thresh:
                low += 1
        return low

    def _is_frame_absent(self, person, conf_thresh):
        low = self._count_body18_low_conf(person, conf_thresh)
        return low >= int(np.ceil(18 * 0.8))

    @staticmethod
    def _all_conf_ge(triplets, indices, conf_thresh):
        for idx in indices:
            c = float(triplets[idx][2]) if idx < len(triplets) and len(triplets[idx]) > 2 else 0.0
            if c < conf_thresh:
                return False
        return True

    def _ref_person_passes_core_rule(self, person, conf_thresh):
        pose = self._body18_triplets(person)
        body_core_ok = self._all_conf_ge(pose, [1, 2, 5, 8, 11], conf_thresh)
        head5_ok = self._all_conf_ge(pose, [0, 14, 15, 16, 17], conf_thresh)
        right_arm_ok = self._all_conf_ge(pose, [2, 3, 4], conf_thresh)
        left_arm_ok = self._all_conf_ge(pose, [5, 6, 7], conf_thresh)
        return body_core_ok and head5_ok and (right_arm_ok or left_arm_ok)

    def _face68_all_conf_ge(self, person, conf_thresh):
        person = self._normalize_person_schema(person)
        face = self._triplets(person.get("face_keypoints_2d", []), 68)
        return self._all_conf_ge(face, list(range(68)), conf_thresh)

    def _video_frame_passes_required_points(self, person, conf_thresh):
        pose = self._body18_triplets(person)
        head6_ok = self._all_conf_ge(pose, [0, 14, 15, 16, 17, 1], conf_thresh)
        shoulder_ok = self._all_conf_ge(pose, [2, 5], conf_thresh)
        arms_ok = self._all_conf_ge(pose, [2, 3, 4, 5, 6, 7], conf_thresh)
        face_ok = self._face68_all_conf_ge(person, conf_thresh)
        return head6_ok and shoulder_ok and arms_ok and face_ok

    def _video_person_passes_trajectory_rule(self, track_frames, conf_thresh):
        if not track_frames:
            return False
        valid_count = 0
        for frame in track_frames:
            people = frame.get("people", [])
            person = people[0] if isinstance(people, list) and len(people) > 0 else self._build_zero_person_openpose()
            if self._video_frame_passes_required_points(person, conf_thresh):
                valid_count += 1
        return (valid_count / float(len(track_frames))) > 0.65

    def _person_sort_x(self, person, conf_thresh):
        pose = self._body18_triplets(person)

        def has(i):
            c = float(pose[i][2]) if i < len(pose) and len(pose[i]) > 2 else 0.0
            return c >= conf_thresh

        # Priority 1: neck x
        if has(1):
            return float(pose[1][0])
        # Priority 2: shoulder midpoint x
        if has(2) and has(5):
            return 0.5 * (float(pose[2][0]) + float(pose[5][0]))
        # Priority 3: mean x of valid body points
        xs = []
        for i in range(18):
            c = float(pose[i][2]) if len(pose[i]) > 2 else 0.0
            if c >= conf_thresh:
                xs.append(float(pose[i][0]))
        if xs:
            return float(np.mean(xs))
        return float("inf")

    def _sorted_people_for_frame(self, frame, conf_thresh):
        people = frame.get("people", [])
        if not isinstance(people, list):
            return []
        normalized = [self._normalize_person_schema(p) for p in people if isinstance(p, dict)]
        normalized.sort(key=lambda p: self._person_sort_x(p, conf_thresh))
        return normalized

    def _extract_person_track(self, frames, person_idx, conf_thresh):
        track = []
        for frame in frames:
            canvas_w = frame.get("canvas_width", 512)
            canvas_h = frame.get("canvas_height", 768)
            people = self._sorted_people_for_frame(frame, conf_thresh)
            if person_idx < len(people):
                person = people[person_idx]
            else:
                person = self._build_zero_person_openpose()
            if self._is_frame_absent(person, conf_thresh):
                person = self._build_zero_person_openpose()
            track.append({
                "people": [person],
                "canvas_width": canvas_w,
                "canvas_height": canvas_h,
            })
        return track

    def _merge_changed_tracks_to_multi_frames(self, changed_tracks, original_frames):
        if not changed_tracks:
            return copy.deepcopy(original_frames)
        n_people = len(changed_tracks)
        n_frames = len(original_frames)
        merged = []
        for fi in range(n_frames):
            src_frame = original_frames[fi] if fi < len(original_frames) else original_frames[0]
            out_frame = {
                "people": [],
                "canvas_width": src_frame.get("canvas_width", 512),
                "canvas_height": src_frame.get("canvas_height", 768),
            }
            for pi in range(n_people):
                person_frame = changed_tracks[pi][fi] if fi < len(changed_tracks[pi]) else changed_tracks[pi][0]
                person_list = person_frame.get("people", [])
                person = person_list[0] if isinstance(person_list, list) and len(person_list) > 0 else self._build_zero_person_openpose()
                out_frame["people"].append(self._normalize_person_schema(person))
            merged.append(out_frame)
        return merged

    def _is_track_frame_valid_for_baseline(self, frame, conf_thresh):
        people = frame.get("people", [])
        if not isinstance(people, list) or len(people) == 0 or not isinstance(people[0], dict):
            return False
        person = self._normalize_person_schema(people[0])
        return not self._is_frame_absent(person, conf_thresh)

    def _ensure_track_first_frame_valid(self, track, conf_thresh):
        if not isinstance(track, list) or len(track) == 0:
            return track
        if self._is_track_frame_valid_for_baseline(track[0], conf_thresh):
            return track

        first_valid_idx = -1
        for i in range(1, len(track)):
            if self._is_track_frame_valid_for_baseline(track[i], conf_thresh):
                first_valid_idx = i
                break

        if first_valid_idx < 0:
            return track

        # Keep frame-0 canvas metadata, replace only person payload.
        src_people = track[first_valid_idx].get("people", [])
        src_person = src_people[0] if isinstance(src_people, list) and len(src_people) > 0 else self._build_zero_person_openpose()
        track0 = dict(track[0])
        track0["people"] = [self._clone_person_fast(src_person)]
        track[0] = track0
        return track

    def _ensure_tracks_first_frame_valid(self, tracks, conf_thresh):
        if not isinstance(tracks, list):
            return tracks
        for i in range(len(tracks)):
            tracks[i] = self._ensure_track_first_frame_valid(tracks[i], conf_thresh)
        return tracks

    def _find_first_full_valid_frame_index(self, frames, n_people, conf_thresh):
        if not isinstance(frames, list) or n_people <= 0:
            return -1
        for fi, frame in enumerate(frames):
            people = self._sorted_people_for_frame(frame, conf_thresh)
            if len(people) < n_people:
                continue
            ok = True
            for pi in range(n_people):
                if self._is_frame_absent(people[pi], conf_thresh):
                    ok = False
                    break
            if ok:
                return fi
        return -1

    def _person_reference_point(self, person, conf_thresh):
        pose = self._body18_triplets(person)

        def conf_ok(i):
            return i < len(pose) and len(pose[i]) > 2 and float(pose[i][2]) >= conf_thresh

        if conf_ok(1):
            return np.array([float(pose[1][0]), float(pose[1][1])], dtype=float)
        if conf_ok(2) and conf_ok(5):
            x = 0.5 * (float(pose[2][0]) + float(pose[5][0]))
            y = 0.5 * (float(pose[2][1]) + float(pose[5][1]))
            return np.array([x, y], dtype=float)
        return None

    @staticmethod
    def _pixel_match_threshold(canvas_w, canvas_h, ratio=0.05):
        return float(ratio) * float(np.sqrt(float(canvas_w) * float(canvas_w) + float(canvas_h) * float(canvas_h)))

    def _stabilize_tracks_after_t_star(self, tracks, frames, full_idx, conf_thresh, ratio=0.05):
        if not isinstance(tracks, list) or len(tracks) == 0:
            return tracks
        if not isinstance(frames, list) or full_idx < 0 or full_idx >= len(frames):
            return tracks

        n = len(tracks)
        id_people = self._sorted_people_for_frame(frames[full_idx], conf_thresh)
        if len(id_people) < n:
            return tracks

        # Lock IDs at t* (left-to-right order at t*).
        last_valid_points = [None] * n
        last_valid_frame_idx = [full_idx] * n
        for pi in range(n):
            p = self._clone_person_fast(id_people[pi])
            tracks[pi][full_idx] = {
                "people": [p],
                "canvas_width": frames[full_idx]["canvas_width"],
                "canvas_height": frames[full_idx]["canvas_height"],
            }
            last_valid_points[pi] = self._person_reference_point(p, conf_thresh)

        # Keep ID order stable for t*+1 ... end by nearest matching to previous valid point.
        for fi in range(full_idx + 1, len(frames)):
            frame = frames[fi]
            canvas_w = frame["canvas_width"]
            canvas_h = frame["canvas_height"]
            tau_px = self._pixel_match_threshold(canvas_w, canvas_h, ratio=ratio)
            candidates = self._sorted_people_for_frame(frame, conf_thresh)
            cand_points = [self._person_reference_point(c, conf_thresh) for c in candidates]
            used = set()

            for pi in range(n):
                ref_pt = last_valid_points[pi]
                best_j = -1
                best_d = float("inf")
                if ref_pt is not None:
                    for cj in range(len(candidates)):
                        if cj in used:
                            continue
                        cp = cand_points[cj]
                        if cp is None:
                            continue
                        d = float(np.sqrt(np.sum((cp - ref_pt) ** 2)))
                        if d < best_d:
                            best_d = d
                            best_j = cj

                gap = fi - last_valid_frame_idx[pi]
                gap_mult = max(1, min(3, gap))
                tau_eff = tau_px * float(gap_mult)
                is_match = (best_j >= 0) and ((n == 1) or (best_d <= tau_eff))
                if is_match:
                    chosen = self._clone_person_fast(candidates[best_j])
                    used.add(best_j)
                    tracks[pi][fi] = {
                        "people": [chosen],
                        "canvas_width": canvas_w,
                        "canvas_height": canvas_h,
                    }
                    new_pt = cand_points[best_j]
                    if new_pt is not None:
                        last_valid_points[pi] = new_pt
                        last_valid_frame_idx[pi] = fi
                else:
                    tracks[pi][fi] = {
                        "people": [self._build_zero_person_openpose()],
                        "canvas_width": canvas_w,
                        "canvas_height": canvas_h,
                    }
        return tracks

    def _stabilize_tracks_before_t_star(self, tracks, frames, full_idx, conf_thresh, ratio=0.05):
        if not isinstance(tracks, list) or len(tracks) == 0:
            return tracks
        if not isinstance(frames, list) or full_idx <= 0 or full_idx >= len(frames):
            return tracks

        n = len(tracks)
        id_people = self._sorted_people_for_frame(frames[full_idx], conf_thresh)
        if len(id_people) < n:
            return tracks

        # Lock IDs at t* and track nearest points backward.
        last_valid_points = [None] * n
        last_valid_frame_idx = [full_idx] * n
        for pi in range(n):
            p = self._clone_person_fast(id_people[pi])
            last_valid_points[pi] = self._person_reference_point(p, conf_thresh)

        # Keep ID order stable for 0 ... t*-1 by reverse nearest matching to next valid point.
        for fi in range(full_idx - 1, -1, -1):
            frame = frames[fi]
            canvas_w = frame["canvas_width"]
            canvas_h = frame["canvas_height"]
            tau_px = self._pixel_match_threshold(canvas_w, canvas_h, ratio=ratio)
            candidates = self._sorted_people_for_frame(frame, conf_thresh)
            cand_points = [self._person_reference_point(c, conf_thresh) for c in candidates]
            used = set()

            for pi in range(n):
                ref_pt = last_valid_points[pi]
                best_j = -1
                best_d = float("inf")
                if ref_pt is not None:
                    for cj in range(len(candidates)):
                        if cj in used:
                            continue
                        cp = cand_points[cj]
                        if cp is None:
                            continue
                        d = float(np.sqrt(np.sum((cp - ref_pt) ** 2)))
                        if d < best_d:
                            best_d = d
                            best_j = cj

                gap = last_valid_frame_idx[pi] - fi
                gap_mult = max(1, min(3, gap))
                tau_eff = tau_px * float(gap_mult)
                is_match = (best_j >= 0) and ((n == 1) or (best_d <= tau_eff))
                if is_match:
                    chosen = self._clone_person_fast(candidates[best_j])
                    used.add(best_j)
                    tracks[pi][fi] = {
                        "people": [chosen],
                        "canvas_width": canvas_w,
                        "canvas_height": canvas_h,
                    }
                    new_pt = cand_points[best_j]
                    if new_pt is not None:
                        last_valid_points[pi] = new_pt
                        last_valid_frame_idx[pi] = fi
                else:
                    tracks[pi][fi] = {
                        "people": [self._build_zero_person_openpose()],
                        "canvas_width": canvas_w,
                        "canvas_height": canvas_h,
                    }
        return tracks

    def _build_anchor_output(self, anchor_people, ref_frame, mode):
        canvas_w = ref_frame.get("canvas_width", 512)
        canvas_h = ref_frame.get("canvas_height", 768)
        if mode == "single_frame_multi_person":
            return [{
                "people": [self._normalize_person_schema(p) for p in anchor_people],
                "canvas_width": canvas_w,
                "canvas_height": canvas_h,
            }]
        if mode == "multi_frame_single_person":
            return [{
                "people": [self._normalize_person_schema(p)],
                "canvas_width": canvas_w,
                "canvas_height": canvas_h,
            } for p in anchor_people]
        raise ValueError(f"Invalid anchor_output_mode: {mode}")

    @staticmethod
    def _validate_anchor_output_shape(anchor_output, n_ref, mode):
        if mode == "single_frame_multi_person":
            if len(anchor_output) != 1:
                raise RuntimeError("single_frame_multi_person must output exactly 1 frame.")
            people = anchor_output[0].get("people", [])
            if len(people) != n_ref:
                raise RuntimeError("single_frame_multi_person people count must equal N_ref.")
            return
        if mode == "multi_frame_single_person":
            if len(anchor_output) != n_ref:
                raise RuntimeError("multi_frame_single_person frame count must equal N_ref.")
            for frame in anchor_output:
                people = frame.get("people", [])
                if len(people) != 1:
                    raise RuntimeError("multi_frame_single_person each frame must contain exactly one person.")
            return
        raise ValueError(f"Invalid anchor_output_mode: {mode}")

    def _passthrough_without_ref(self, pose_keypoint, anchor_output_mode, conf_thresh):
        changed_output = copy.deepcopy(pose_keypoint)
        if not changed_output:
            return (pose_keypoint, pose_keypoint)

        first_frame = changed_output[0]
        sorted_people = self._sorted_people_for_frame(first_frame, conf_thresh)
        anchor_output = self._build_anchor_output(sorted_people, first_frame, anchor_output_mode)
        self._validate_anchor_output_shape(anchor_output, len(sorted_people), anchor_output_mode)
        return (changed_output, anchor_output)

    def process(self, pose_keypoint, ref_pose_keypoint=None, manual_anchor_pose=None, alignment_mode=False, hand_scaling=True, foot_scaling=True, offset_stabilizer=True, offset_stabilizer_x=False, best_hand_search=True, use_shoulder_fk_for_hand=False, best_neck_search=False, final_offset_alignment=True, first_frame_offset_alignment=False, confidence_threshold=0.30, output_absolute_coordinates=True, anchor_output_mode="single_frame_multi_person", print_detailed_logs=False):
        if not pose_keypoint or len(pose_keypoint) == 0:
            return (pose_keypoint, pose_keypoint)

        runtime_cfg = self._resolve_runtime_config(
            alignment_mode=alignment_mode,
            hand_scaling=hand_scaling,
            foot_scaling=foot_scaling,
            offset_stabilizer=offset_stabilizer,
            offset_stabilizer_x=offset_stabilizer_x,
            best_hand_search=best_hand_search,
            use_shoulder_fk_for_hand=use_shoulder_fk_for_hand,
            best_neck_search=best_neck_search,
            final_offset_alignment=final_offset_alignment,
            first_frame_offset_alignment=first_frame_offset_alignment,
            anchor_output_mode=anchor_output_mode,
            print_detailed_logs=print_detailed_logs,
            confidence_threshold=confidence_threshold,
            output_absolute_coordinates=output_absolute_coordinates,
        )
        self._assert_single_frame_input("ref_pose_keypoint", ref_pose_keypoint)
        self._assert_single_frame_input("manual_anchor_pose", manual_anchor_pose)
        if runtime_cfg.anchor_output_mode not in ("single_frame_multi_person", "multi_frame_single_person"):
            raise ValueError(f"Invalid anchor_output_mode: {runtime_cfg.anchor_output_mode}")
        if ref_pose_keypoint is None or len(ref_pose_keypoint) == 0:
            max_people = 0
            for frame in pose_keypoint:
                people = frame.get("people", [])
                if isinstance(people, list):
                    max_people = max(max_people, len(people))
            if max_people <= 1:
                return self._process_single(
                    pose_keypoint=pose_keypoint,
                    ref_pose_keypoint=ref_pose_keypoint,
                    manual_anchor_pose=manual_anchor_pose,
                    config=runtime_cfg,
                )
            self._log_multi_summary(
                "no_ref_passthrough",
                input_frames=len(pose_keypoint),
                input_max_people=max_people,
                anchor_output_mode=runtime_cfg.anchor_output_mode,
            )
            return self._passthrough_without_ref(
                pose_keypoint=pose_keypoint,
                anchor_output_mode=runtime_cfg.anchor_output_mode,
                conf_thresh=float(runtime_cfg.confidence_threshold),
            )

        conf_thresh = float(runtime_cfg.confidence_threshold)
        ref_frame = ref_pose_keypoint[0]
        ref_people = ref_frame.get("people", [])
        ref_people_filtered = []
        for person in ref_people:
            if isinstance(person, dict) and self._ref_person_passes_core_rule(person, conf_thresh):
                ref_people_filtered.append(self._normalize_person_schema(person))
        # Sort filtered reference people from left to right.
        ref_people_filtered.sort(key=lambda p: self._person_sort_x(p, conf_thresh))
        n_ref = len(ref_people_filtered)
        self._log_multi_summary(
            "ref_filter_done",
            ref_input_people=len(ref_people) if isinstance(ref_people, list) else 0,
            ref_filtered_people=n_ref,
            conf_thresh=conf_thresh,
        )
        if n_ref == 0:
            self._log_multi_summary(
                "ref_empty_passthrough",
                changed_policy="passthrough",
                anchor_policy="empty_list",
                input_frames=len(pose_keypoint),
            )
            return (pose_keypoint, [])

        max_people = 0
        for frame in pose_keypoint:
            people = frame.get("people", [])
            if isinstance(people, list):
                max_people = max(max_people, len(people))

        candidate_tracks = []
        if max_people <= 1:
            # Single-person video path: skip trajectory filtering.
            track = self._extract_person_track(pose_keypoint, 0, conf_thresh)
            candidate_tracks.append(track)
        else:
            for pi in range(max_people):
                track = self._extract_person_track(pose_keypoint, pi, conf_thresh)
                if self._video_person_passes_trajectory_rule(track, conf_thresh):
                    candidate_tracks.append(track)
        valid_tracks_before_resize = len(candidate_tracks)
        trim_count = 0
        pad_count = 0
        pad_source = "none"

        if len(candidate_tracks) > n_ref:
            trim_count = len(candidate_tracks) - n_ref
            candidate_tracks = candidate_tracks[:n_ref]
        elif len(candidate_tracks) < n_ref:
            pad_count = n_ref - len(candidate_tracks)
            if len(candidate_tracks) == 0:
                zero_track = self._extract_person_track([{
                    "people": [self._build_zero_person_openpose()],
                    "canvas_width": f.get("canvas_width", 512),
                    "canvas_height": f.get("canvas_height", 768),
                } for f in pose_keypoint], 0, conf_thresh)
                candidate_tracks = [self._clone_track_fast(zero_track) for _ in range(n_ref)]
                pad_source = "zero_track"
            else:
                src_track = candidate_tracks[-1]
                while len(candidate_tracks) < n_ref:
                    candidate_tracks.append(self._clone_track_fast(src_track))
                pad_source = "last_valid_track"

        full_valid_frame_idx = self._find_first_full_valid_frame_index(
            pose_keypoint, len(candidate_tracks), conf_thresh
        )
        candidate_tracks = self._stabilize_tracks_before_t_star(
            candidate_tracks, pose_keypoint, full_valid_frame_idx, conf_thresh, ratio=0.08
        )
        candidate_tracks = self._stabilize_tracks_after_t_star(
            candidate_tracks, pose_keypoint, full_valid_frame_idx, conf_thresh, ratio=0.08
        )

        manual_people_filtered = []
        if manual_anchor_pose is not None and len(manual_anchor_pose) > 0:
            manual_people = manual_anchor_pose[0].get("people", [])
            for person in manual_people:
                if isinstance(person, dict) and self._ref_person_passes_core_rule(person, conf_thresh):
                    manual_people_filtered.append(self._normalize_person_schema(person))
            if len(manual_people_filtered) > n_ref:
                manual_people_filtered = manual_people_filtered[:n_ref]

        candidate_tracks = self._ensure_tracks_first_frame_valid(candidate_tracks, conf_thresh)
        self._log_multi_summary(
            "video_tracks_ready",
            input_max_people=max_people,
            valid_tracks_before_resize=valid_tracks_before_resize,
            final_tracks=len(candidate_tracks),
            n_ref=n_ref,
            full_valid_frame_idx=full_valid_frame_idx,
            trim_count=trim_count,
            pad_count=pad_count,
            pad_source=pad_source,
            manual_filtered_people=len(manual_people_filtered),
        )

        changed_tracks = []
        anchor_people = []
        for i in range(n_ref):
            ref_single_frame = {
                "people": [self._clone_person_fast(ref_people_filtered[i])],
                "canvas_width": ref_frame.get("canvas_width", 512),
                "canvas_height": ref_frame.get("canvas_height", 768),
            }
            manual_single = None
            if i < len(manual_people_filtered):
                manual_single = [{
                    "people": [self._clone_person_fast(manual_people_filtered[i])],
                    "canvas_width": ref_frame.get("canvas_width", 512),
                    "canvas_height": ref_frame.get("canvas_height", 768),
                }]

            # Multi-person mode keeps single-person verbose logs muted by default.
            if runtime_cfg.print_detailed_logs:
                changed_i, anchor_i = self._process_single(
                    pose_keypoint=candidate_tracks[i],
                    ref_pose_keypoint=[ref_single_frame],
                    manual_anchor_pose=manual_single,
                    config=runtime_cfg,
                )
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    changed_i, anchor_i = self._process_single(
                        pose_keypoint=candidate_tracks[i],
                        ref_pose_keypoint=[ref_single_frame],
                        manual_anchor_pose=manual_single,
                        config=runtime_cfg,
                    )
            changed_tracks.append(changed_i)
            if anchor_i and len(anchor_i) > 0:
                a_people = anchor_i[0].get("people", [])
                if isinstance(a_people, list) and len(a_people) > 0:
                    anchor_people.append(self._normalize_person_schema(a_people[0]))
                else:
                    anchor_people.append(self._build_zero_person_openpose())
            else:
                anchor_people.append(self._build_zero_person_openpose())

        changed_output = self._merge_changed_tracks_to_multi_frames(changed_tracks, pose_keypoint)
        anchor_output = self._build_anchor_output(anchor_people, ref_frame, runtime_cfg.anchor_output_mode)
        self._validate_anchor_output_shape(anchor_output, n_ref, runtime_cfg.anchor_output_mode)
        self._log_multi_summary(
            "done",
            changed_frames=len(changed_output),
            changed_people_per_frame=(len(changed_output[0].get("people", [])) if len(changed_output) > 0 else 0),
            anchor_frames=len(anchor_output),
            anchor_mode=runtime_cfg.anchor_output_mode,
            n_ref=n_ref,
        )
        return (changed_output, anchor_output)

    def _process_single(self, pose_keypoint, ref_pose_keypoint=None, manual_anchor_pose=None, alignment_mode=False, hand_scaling=True, foot_scaling=True, offset_stabilizer=True, offset_stabilizer_x=False, best_hand_search=True, use_shoulder_fk_for_hand=False, best_neck_search=False, final_offset_alignment=True, first_frame_offset_alignment=False, confidence_threshold=0.30, output_absolute_coordinates=True, config=None):
        if not pose_keypoint or len(pose_keypoint) == 0:
            return (pose_keypoint, pose_keypoint) # Return tuple for both outputs
        
        frame_data = pose_keypoint[0]
        canvas_width = frame_data.get('canvas_width', 512)
        canvas_height = frame_data.get('canvas_height', 768)
        
        batch_pose_data = self.parse_keypoints(pose_keypoint, canvas_width, canvas_height)
        
        ref_data = None
        ref_canvas_width, ref_canvas_height = canvas_width, canvas_height
        
        if ref_pose_keypoint is not None:
            ref_frame_data = ref_pose_keypoint[0]
            ref_canvas_width = ref_frame_data.get('canvas_width', 512)
            ref_canvas_height = ref_frame_data.get('canvas_height', 768)
            
            ref_batch = self.parse_keypoints(ref_pose_keypoint, ref_canvas_width, ref_canvas_height)
            if ref_batch:
                ref_data = ref_batch[0]
        
        # Process manual override data if provided
        manual_ref_data = None
        if manual_anchor_pose is not None:
            manual_frame_data = manual_anchor_pose[0]
            manual_canvas_width = manual_frame_data.get('canvas_width', 512)
            manual_canvas_height = manual_frame_data.get('canvas_height', 768)
            
            manual_batch = self.parse_keypoints(manual_anchor_pose, manual_canvas_width, manual_canvas_height)
            if manual_batch:
                manual_ref_data = manual_batch[0]

        runtime_config = self._resolve_runtime_config(
            config=config,
            alignment_mode=alignment_mode,
            hand_scaling=hand_scaling,
            foot_scaling=foot_scaling,
            offset_stabilizer=offset_stabilizer,
            offset_stabilizer_x=offset_stabilizer_x,
            best_hand_search=best_hand_search,
            use_shoulder_fk_for_hand=use_shoulder_fk_for_hand,
            best_neck_search=best_neck_search,
            final_offset_alignment=final_offset_alignment,
            first_frame_offset_alignment=first_frame_offset_alignment,
            confidence_threshold=confidence_threshold,
            output_absolute_coordinates=output_absolute_coordinates,
        )
        
        # Capture anchor_idx from the main processing function
        processed_batch, anchor_idx, best_hand = self.apply_batch_proportion_changes(
            batch_pose_data, ref_data,
            canvas_width, canvas_height, ref_canvas_width, ref_canvas_height,
            manual_ref_data=manual_ref_data,
            config=runtime_config
        )
        
        result_frames = self.serialize_to_sdpose(processed_batch, pose_keypoint)
        
        # --- Extract the original unpolluted anchor pose keypoint ---
        if manual_anchor_pose is not None:
            # Echo logic: if manual anchor was provided, output it exactly as is
            anchor_output = manual_anchor_pose
        else:
            # Automatic logic: extract the original frame selected by WSCS
            safe_anchor_idx = anchor_idx if 0 <= anchor_idx < len(pose_keypoint) else 0
            anchor_output = [copy.deepcopy(pose_keypoint[safe_anchor_idx])]

            # Inject best-hand hand into anchor output so the anchor stream reflects hand selection.
            if best_hand is not None and len(anchor_output) > 0:
                frame = anchor_output[0]
                people = frame.get("people", [])
                if isinstance(people, list) and len(people) > 0 and isinstance(people[0], dict):
                    person = people[0]
                    side = best_hand.get("side")
                    hand_pts = best_hand.get("pts")
                    hand_conf = best_hand.get("conf")
                    if hand_pts is not None and hand_conf is not None:
                        pose_2d = person.get("pose_keypoints_2d", [])
                        canvas_w = float(frame.get("canvas_width", 512))
                        canvas_h = float(frame.get("canvas_height", 768))

                        def _read_body_pt(kpt_idx):
                            base = kpt_idx * 3
                            if len(pose_2d) >= base + 3:
                                x = float(pose_2d[base])
                                y = float(pose_2d[base + 1])
                                c = float(pose_2d[base + 2])
                                return np.array([x, y], dtype=float), c
                            return None, 0.0

                        def _is_normalized_domain():
                            vals = []
                            for key in ("pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d", "foot_keypoints_2d"):
                                arr = person.get(key, [])
                                if isinstance(arr, list) and len(arr) >= 3:
                                    vals.extend([abs(float(v)) for v in arr[0:90:3]])
                                    vals.extend([abs(float(v)) for v in arr[1:91:3]])
                            vals = [v for v in vals if v > 1e-6]
                            if not vals:
                                return False
                            return max(vals) <= 1.5

                        def _convert_triplets_to_pixel(arr):
                            if not isinstance(arr, list) or len(arr) < 3:
                                return arr
                            out = arr[:]
                            for i in range(0, len(out), 3):
                                if i + 2 < len(out):
                                    out[i] = float(out[i]) * canvas_w
                                    out[i + 1] = float(out[i + 1]) * canvas_h
                            return out

                        def _mirror_by_root_x(pts):
                            out = pts.copy()
                            root_x = float(out[0][0])
                            out[:, 0] = 2.0 * root_x - out[:, 0]
                            return out

                        def _bind_to_wrist(pts, wrist_xy):
                            out = pts.copy()
                            if wrist_xy is not None:
                                out = out + (wrist_xy - out[0])
                            return out

                        def _to_hand_flat(pts, conf):
                            out = []
                            total = 21
                            for i in range(total):
                                if i < len(pts):
                                    x = float(pts[i][0])
                                    y = float(pts[i][1])
                                    c = float(conf[i]) if i < len(conf) else 0.0
                                    if abs(x) < 1e-6 and abs(y) < 1e-6:
                                        out.extend([0.0, 0.0, 0.0])
                                    else:
                                        out.extend([x, y, c])
                                else:
                                    out.extend([0.0, 0.0, 0.0])
                            return out

                        def _hand_root_xy(hand_flat):
                            if not isinstance(hand_flat, list) or len(hand_flat) < 3:
                                return None, 0.0
                            x = float(hand_flat[0])
                            y = float(hand_flat[1])
                            c = float(hand_flat[2])
                            if c < 0.05:
                                return None, c
                            return np.array([x, y], dtype=float), c

                        def _dist(a, b):
                            if a is None or b is None:
                                return None
                            return float(np.sqrt(np.sum((a - b) ** 2)))

                        def _detect_swapped_hand_keys():
                            # Returns True when source frame uses swapped hand keys:
                            # real-left in hand_right_keypoints_2d and real-right in hand_left_keypoints_2d.
                            left_key_root, _ = _hand_root_xy(person.get("hand_left_keypoints_2d", []))
                            right_key_root, _ = _hand_root_xy(person.get("hand_right_keypoints_2d", []))
                            if left_key_root is None or right_key_root is None:
                                return True  # Keep legacy-safe fallback for this project.

                            d_ll = _dist(left_key_root, left_wrist_xy)
                            d_lr = _dist(left_key_root, right_wrist_xy)
                            d_rr = _dist(right_key_root, right_wrist_xy)
                            d_rl = _dist(right_key_root, left_wrist_xy)
                            if None in (d_ll, d_lr, d_rr, d_rl):
                                return True

                            normal_score = d_ll + d_rr
                            swapped_score = d_lr + d_rl
                            return swapped_score < normal_score

                        # Normalize legacy input to pixel domain before anchor-hand injection.
                        if _is_normalized_domain():
                            for key in ("pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d", "foot_keypoints_2d"):
                                person[key] = _convert_triplets_to_pixel(person.get(key, []))
                            pose_2d = person.get("pose_keypoints_2d", [])

                        template = np.asarray(hand_pts, dtype=float).copy()
                        conf_arr = np.asarray(hand_conf, dtype=float).copy()
                        left_wrist_xy, left_wrist_conf = _read_body_pt(7)   # left wrist
                        right_wrist_xy, right_wrist_conf = _read_body_pt(4) # right wrist
                        if left_wrist_conf < 0.05:
                            left_wrist_xy = None
                        if right_wrist_conf < 0.05:
                            right_wrist_xy = None

                        if side == "left":
                            left_hand_pts = _bind_to_wrist(template, left_wrist_xy)
                            right_hand_pts = _bind_to_wrist(_mirror_by_root_x(template), right_wrist_xy)
                        elif side == "right":
                            right_hand_pts = _bind_to_wrist(template, right_wrist_xy)
                            left_hand_pts = _bind_to_wrist(_mirror_by_root_x(template), left_wrist_xy)
                        else:
                            left_hand_pts = _bind_to_wrist(template, left_wrist_xy)
                            right_hand_pts = _bind_to_wrist(_mirror_by_root_x(template), right_wrist_xy)

                        left_flat = _to_hand_flat(left_hand_pts, conf_arr)
                        right_flat = _to_hand_flat(right_hand_pts, conf_arr)
                        if _detect_swapped_hand_keys():
                            person["hand_left_keypoints_2d"] = right_flat
                            person["hand_right_keypoints_2d"] = left_flat
                        else:
                            person["hand_left_keypoints_2d"] = left_flat
                            person["hand_right_keypoints_2d"] = right_flat

        def apply_best_neck_to_anchor_output(anchor_frames):
            if not best_neck_search:
                return
            if not isinstance(anchor_frames, list) or len(anchor_frames) == 0:
                return
            frame = anchor_frames[0]
            people = frame.get("people", [])
            if not isinstance(people, list) or len(people) == 0 or not isinstance(people[0], dict):
                return
            person = people[0]
            pose_2d = person.get("pose_keypoints_2d", [])
            if not isinstance(pose_2d, list) or len(pose_2d) < 18 * 3:
                return

            canvas_w = float(frame.get("canvas_width", 512))
            canvas_h = float(frame.get("canvas_height", 768))
            conf_thresh_local = float(confidence_threshold)
            def _pt_exists(pt):
                try:
                    return float(np.sum(np.abs(pt))) > 0.01
                except Exception:
                    return False

            def _dist(p1, p2):
                if not (_pt_exists(p1) and _pt_exists(p2)):
                    return None
                return float(np.sqrt(np.sum((p1 - p2) ** 2)))

            def _is_normalized_domain_local():
                vals = []
                for key in ("pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d", "foot_keypoints_2d"):
                    arr = person.get(key, [])
                    if isinstance(arr, list) and len(arr) >= 3:
                        vals.extend([abs(float(v)) for v in arr[0:90:3]])
                        vals.extend([abs(float(v)) for v in arr[1:91:3]])
                vals = [v for v in vals if v > 1e-6]
                if not vals:
                    return False
                return max(vals) <= 1.5

            def _convert_triplets_to_pixel_local(arr):
                if not isinstance(arr, list) or len(arr) < 3:
                    return arr
                out = arr[:]
                for i in range(0, len(out), 3):
                    if i + 2 < len(out):
                        out[i] = float(out[i]) * canvas_w
                        out[i + 1] = float(out[i + 1]) * canvas_h
                return out

            # Keep anchor output in pixel domain when applying geometric neck replacement.
            if _is_normalized_domain_local():
                for key in ("pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d", "foot_keypoints_2d"):
                    person[key] = _convert_triplets_to_pixel_local(person.get(key, []))
                pose_2d = person.get("pose_keypoints_2d", [])

            def _read_xy_conf(flat, idx):
                base = idx * 3
                if not isinstance(flat, list) or len(flat) < base + 3:
                    return None, 0.0
                x = float(flat[base])
                y = float(flat[base + 1])
                c = float(flat[base + 2])
                if abs(x) < 1e-6 and abs(y) < 1e-6:
                    return None, c
                return np.array([x, y], dtype=float), c

            anc_sh_l, anc_sh_l_conf = _read_xy_conf(pose_2d, 2)
            anc_sh_r, anc_sh_r_conf = _read_xy_conf(pose_2d, 5)
            anc_neck_xy, _ = _read_xy_conf(pose_2d, 1)
            anc_nose_xy, _ = _read_xy_conf(pose_2d, 0)
            if anc_sh_l is None or anc_sh_r is None:
                return
            if anc_sh_l_conf < conf_thresh_local or anc_sh_r_conf < conf_thresh_local:
                return
            if anc_neck_xy is None or anc_nose_xy is None:
                return

            anchor_shoulder_len = float(np.sqrt(np.sum((anc_sh_l - anc_sh_r) ** 2)))
            if anchor_shoulder_len <= 1e-6:
                return

            best_neck_len = -1.0
            best_neck_xy = None
            best_nose_xy = None
            for parsed_frame in batch_pose_data:
                c = parsed_frame['bodies']['candidate']
                conf = parsed_frame['bodies']['candidate_conf']
                if not (_pt_exists(c[2]) and _pt_exists(c[5]) and _pt_exists(c[1]) and _pt_exists(c[0])):
                    continue
                if len(conf) <= 5:
                    continue
                if conf[2] < conf_thresh_local or conf[5] < conf_thresh_local or conf[1] < conf_thresh_local or conf[0] < conf_thresh_local:
                    continue
                shoulder_len = _dist(c[2], c[5])
                if shoulder_len is None or shoulder_len <= 1e-6:
                    continue
                if abs(shoulder_len - anchor_shoulder_len) / anchor_shoulder_len > 0.05:
                    continue
                neck_len = _dist(c[0], c[1])
                if neck_len is None or neck_len <= 1e-6:
                    continue
                if neck_len > best_neck_len:
                    best_neck_len = float(neck_len)
                    best_neck_xy = c[1].copy()
                    best_nose_xy = c[0].copy()

            if best_neck_xy is None or best_nose_xy is None:
                return

            # Keep original neck anchor position; transplant only best neck vector.
            best_neck_vec = best_nose_xy - best_neck_xy
            new_neck_xy = anc_neck_xy.copy()
            new_nose_xy = new_neck_xy + best_neck_vec
            delta = new_nose_xy - anc_nose_xy

            def _shift_xy_in_triplets(flat, idx, shift):
                base = idx * 3
                if not isinstance(flat, list) or len(flat) < base + 3:
                    return
                x = float(flat[base])
                y = float(flat[base + 1])
                c = float(flat[base + 2])
                if c < conf_thresh_local:
                    return
                flat[base] = x + float(shift[0])
                flat[base + 1] = y + float(shift[1])

            # Keep neck at original anchor location; move head/face with transplanted neck vector.
            pose_2d[3] = float(new_neck_xy[0])
            pose_2d[4] = float(new_neck_xy[1])
            _shift_xy_in_triplets(pose_2d, 0, delta)   # nose
            _shift_xy_in_triplets(pose_2d, 14, delta)  # right eye
            _shift_xy_in_triplets(pose_2d, 15, delta)  # left eye
            _shift_xy_in_triplets(pose_2d, 16, delta)  # right ear
            _shift_xy_in_triplets(pose_2d, 17, delta)  # left ear
            person["pose_keypoints_2d"] = pose_2d

            face_2d = person.get("face_keypoints_2d", [])
            if isinstance(face_2d, list) and len(face_2d) >= 3:
                for i in range(0, len(face_2d), 3):
                    if i + 2 >= len(face_2d):
                        break
                    x = float(face_2d[i])
                    y = float(face_2d[i + 1])
                    c = float(face_2d[i + 2])
                    if c < conf_thresh_local:
                        continue
                    face_2d[i] = x + float(delta[0])
                    face_2d[i + 1] = y + float(delta[1])
                person["face_keypoints_2d"] = face_2d

        def enforce_anchor_output_domain(anchor_frames):
            if not isinstance(anchor_frames, list):
                return

            to_absolute = bool(output_absolute_coordinates)

            def _is_person_normalized(p):
                vals = []
                for key in ("pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d", "foot_keypoints_2d"):
                    arr = p.get(key, [])
                    if isinstance(arr, list) and len(arr) >= 3:
                        vals.extend([abs(float(v)) for v in arr[0:120:3]])
                        vals.extend([abs(float(v)) for v in arr[1:121:3]])
                vals = [v for v in vals if v > 1e-6]
                if not vals:
                    return False
                return max(vals) <= 1.5

            def _convert_triplets(arr, canvas_w, canvas_h, to_abs):
                if not isinstance(arr, list) or len(arr) < 3:
                    return arr
                out = arr[:]
                for i in range(0, len(out), 3):
                    if i + 2 >= len(out):
                        break
                    x = float(out[i])
                    y = float(out[i + 1])
                    if to_abs:
                        out[i] = x * canvas_w
                        out[i + 1] = y * canvas_h
                    else:
                        out[i] = x / canvas_w if canvas_w > 1e-6 else x
                        out[i + 1] = y / canvas_h if canvas_h > 1e-6 else y
                return out

            for frame in anchor_frames:
                if not isinstance(frame, dict):
                    continue
                canvas_w = float(frame.get("canvas_width", 512))
                canvas_h = float(frame.get("canvas_height", 768))
                people = frame.get("people", [])
                if not isinstance(people, list):
                    continue
                for person in people:
                    if not isinstance(person, dict):
                        continue
                    is_norm = _is_person_normalized(person)
                    if to_absolute and is_norm:
                        for key in ("pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d", "foot_keypoints_2d"):
                            person[key] = _convert_triplets(person.get(key, []), canvas_w, canvas_h, True)
                    elif (not to_absolute) and (not is_norm):
                        for key in ("pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d", "foot_keypoints_2d"):
                            person[key] = _convert_triplets(person.get(key, []), canvas_w, canvas_h, False)

        # Apply optional best-neck replacement on anchor output only.
        apply_best_neck_to_anchor_output(anchor_output)
        # Enforce anchor output coordinate domain by node parameter.
        enforce_anchor_output_domain(anchor_output)
        
        return (result_frames, anchor_output) # Output both data streams

    # ========== Phase 0: Coordinate Conversion Functions ==========
    def convert_to_physical_coords(self, pose_data, width, height):
        """Convert normalized coordinates (0-1) to physical pixel coordinates"""
        sx = float(width)
        sy = float(height)

        def _scale_points_inplace(points):
            if not isinstance(points, np.ndarray) or points.size == 0:
                return
            mask = matrix_ops_external.valid_point_mask(points, eps=0.01)
            points[..., 0][mask] *= sx
            points[..., 1][mask] *= sy

        for frame in pose_data:
            _scale_points_inplace(frame['bodies']['candidate'])
            _scale_points_inplace(frame['faces'])
            _scale_points_inplace(frame['hands'])
            _scale_points_inplace(frame['feet'])
        return pose_data

    def convert_to_normalized_coords(self, pose_data, width, height):
        """Convert physical pixel coordinates back to normalized (0-1)"""
        sx = float(width)
        sy = float(height)

        def _scale_points_inplace(points):
            if not isinstance(points, np.ndarray) or points.size == 0:
                return
            mask = matrix_ops_external.valid_point_mask(points, eps=0.01)
            points[..., 0][mask] /= sx
            points[..., 1][mask] /= sy

        for frame in pose_data:
            _scale_points_inplace(frame['bodies']['candidate'])
            _scale_points_inplace(frame['faces'])
            _scale_points_inplace(frame['hands'])
            _scale_points_inplace(frame['feet'])
        return pose_data

    # ========== Phase 1: 13-Bone Global RPCA Calculation ==========
    def calculate_13_bone_global_rpca(self, anchor_candidate, anchor_faces, f0_candidate, f0_faces, alignment_mode=True):
        """Calculate 13-bone Global RPCA multiplier (Anchor vs F0, deviation-based sorting)."""
        return calculate_global_rpca_external(
            anchor_candidate=anchor_candidate,
            anchor_faces=anchor_faces,
            f0_candidate=f0_candidate,
            f0_faces=f0_faces,
            alignment_mode=alignment_mode,
        )

    # ========== Phase 2: FK Value Extraction Functions ==========
    def _lr_complement(self, left, right):
        return scale_solver_external.lr_complement(left, right)

    def _avg_ratio_two_sides(self, ref_l, anc_l, ref_r, anc_r):
        return scale_solver_external.avg_ratio_two_sides(ref_l, anc_l, ref_r, anc_r)

    def extract_fk_values_part1_torso_neck(self, ref_candidate, anc_candidate):
        """Extract torso and neck FK values."""
        return scale_solver_external.extract_fk_values_part1_torso_neck(ref_candidate, anc_candidate)

    def extract_fk_values_part2_arms(self, ref_candidate, anc_candidate):
        """Extract arm FK values with mirror compensation and fallback."""
        return scale_solver_external.extract_fk_values_part2_arms(ref_candidate, anc_candidate)

    def extract_fk_values_part3_legs(self, ref_candidate, anc_candidate):
        """Extract leg FK values with mirror compensation and fallback."""
        return scale_solver_external.extract_fk_values_part3_legs(ref_candidate, anc_candidate)

    def extract_fk_values_part4_hands_feet_face(self, ref_candidate, ref_faces, ref_hands, ref_feet, anc_candidate, anc_faces, anc_hands, anc_feet, anc_hand_baseline=None):
        """Extract hand, foot, and face FK values."""
        return scale_solver_external.extract_fk_values_part4_hands_feet_face(
            ref_candidate, ref_faces, ref_hands, ref_feet, anc_candidate, anc_faces, anc_hands, anc_feet, anc_hand_baseline=anc_hand_baseline
        )

    def validate_hand_fk(self, hand_fk, torso, neck, upper_arm, lower_arm, upper_leg, lower_leg, foot_edge1, foot_edge2, foot_edge3, body_to_foot_ankle, face_x, face_y, eye_width=None):
        """Validate hand FK value to prevent anomalies from dense keypoint measurements."""
        return scale_solver_external.validate_hand_fk(
            hand_fk, torso, neck, upper_arm, lower_arm, upper_leg, lower_leg, foot_edge1, foot_edge2, foot_edge3, body_to_foot_ankle, face_x, face_y, eye_width=eye_width, logger=print
        )

    # ========== Phase 3: Forge Final Scale Constants ==========
    def forge_final_scale_constants(self, fk_values, global_rpca_multiplier, alignment_mode=True):
        """Forge final scaling constants: FK × Global RPCA."""
        return forge_final_scales_external(
            fk_values=fk_values,
            global_rpca_multiplier=global_rpca_multiplier,
            alignment_mode=alignment_mode,
        )

    def apply_batch_proportion_changes(self, batch_pose_data, ref_data,
                                     canvas_width, canvas_height, ref_canvas_width, ref_canvas_height, manual_ref_data=None, config=None):
        # NOTE: This complex calculation logic is preserved exactly as requested.
        # Logic updated to fix video offset drift by using a fixed global offset from an anchor frame.
        
        if not batch_pose_data or not ref_data:
            return batch_pose_data, 0, None

        results_vis = batch_pose_data
        runtime_cfg = self._resolve_runtime_config(config=config)
        alignment_mode = runtime_cfg.alignment_mode
        hand_scaling = runtime_cfg.hand_scaling
        foot_scaling = runtime_cfg.foot_scaling
        offset_stabilizer = runtime_cfg.offset_stabilizer
        offset_stabilizer_x = runtime_cfg.offset_stabilizer_x
        best_hand_search = runtime_cfg.best_hand_search
        use_shoulder_fk_for_hand = runtime_cfg.use_shoulder_fk_for_hand
        best_neck_search = runtime_cfg.best_neck_search
        final_offset_alignment = runtime_cfg.final_offset_alignment
        first_frame_offset_alignment = runtime_cfg.first_frame_offset_alignment
        output_absolute_coordinates = runtime_cfg.output_absolute_coordinates
        conf_thresh = runtime_cfg.confidence_threshold

        # --- Helper for SAFE ADDITION (Fixes Tensor Broadcasting Crash) ---
        def safe_add(array, offset):
            """
            Adds offset ONLY to points that are not (0,0).
            Strict threshold to avoid moving noise.
            """
            matrix_ops_external.masked_add(array, offset, eps=0.01)

        def has_pt(pt):
            return np.sum(np.abs(pt)) > 0.01

        # Original normalized distance (Used by Main RPCA loop)
        def get_dist(p1, p2):
            return np.sqrt(np.sum((p1 - p2)**2))

        # --- Phase 0.5: Unified Bidirectional Interpolation Function ---
        def apply_bidirectional_interpolation(raw_offsets):
            """
            Apply bidirectional interpolation to fill missing offsets.

            Args:
                raw_offsets: List[Optional[np.ndarray]] - Raw offset array, None indicates missing

            Returns:
                final_offsets: List[np.ndarray] - Filled offset array, no None
            """
            final_offsets = []
            valid_indices = [i for i, x in enumerate(raw_offsets) if x is not None]

            if not valid_indices:
                # Fallback: All zeros if all frames are missing
                final_offsets = [np.array([0.0, 0.0])] * len(raw_offsets)
            else:
                final_offsets = [None] * len(raw_offsets)

                # Backfill (Start)
                first_idx = valid_indices[0]
                first_val = raw_offsets[first_idx]
                for i in range(first_idx + 1):
                    final_offsets[i] = first_val.copy()

                # Forward fill (End)
                last_idx = valid_indices[-1]
                last_val = raw_offsets[last_idx]
                for i in range(last_idx, len(raw_offsets)):
                    final_offsets[i] = last_val.copy()

                # Lerp (Middle)
                for k in range(len(valid_indices) - 1):
                    idx_a = valid_indices[k]
                    idx_b = valid_indices[k+1]
                    val_a = raw_offsets[idx_a]
                    val_b = raw_offsets[idx_b]

                    dist = idx_b - idx_a
                    if dist > 1:
                        for j in range(1, dist):
                            w = j / float(dist)
                            interp_val = val_a * (1.0 - w) + val_b * w
                            final_offsets[idx_a + j] = interp_val

                    # Ensure anchor points are set
                    final_offsets[idx_a] = val_a
                    final_offsets[idx_b] = val_b

            return final_offsets

        def build_frame_geom_cache(frames):
            """
            Cache per-frame body geometry used repeatedly by offset/stabilizer paths.
            """
            cache = []
            for frame in frames:
                c = frame['bodies']['candidate']
                if isinstance(c, np.ndarray) and c.ndim == 2 and c.shape[1] >= 2:
                    present = matrix_ops_external.valid_point_mask(c, eps=0.01)
                else:
                    present = np.zeros((18,), dtype=bool)

                def pt_or_none(idx):
                    if idx < len(c) and idx < len(present) and present[idx]:
                        return c[idx]
                    return None

                p1 = pt_or_none(1)   # neck
                p0 = pt_or_none(0)   # nose
                p2 = pt_or_none(2)   # right shoulder
                p5 = pt_or_none(5)   # left shoulder
                p8 = pt_or_none(8)   # right hip
                p11 = pt_or_none(11) # left hip

                mid_sh = ((p2 + p5) * 0.5) if (p2 is not None and p5 is not None) else None
                mid_hip = ((p8 + p11) * 0.5) if (p8 is not None and p11 is not None) else None

                cache.append({
                    "candidate": c,
                    "neck": p1,
                    "nose": p0,
                    "r_shoulder": p2,
                    "l_shoulder": p5,
                    "mid_shoulder": mid_sh,
                    "mid_hip": mid_hip,
                })
            return cache

        def build_stabilizer_point_cache(frames):
            """
            Cache per-frame stabilizer base point:
            priority neck -> shoulder midpoint (both gated by confidence threshold).
            """
            cache = []
            for frame in frames:
                c = frame['bodies']['candidate']
                conf = frame['bodies']['candidate_conf']

                def has_valid_pt(idx):
                    return idx < len(conf) and has_pt(c[idx]) and conf[idx] >= conf_thresh

                if has_valid_pt(1):
                    cache.append(c[1].copy())
                elif has_valid_pt(2) and has_valid_pt(5):
                    cache.append(((c[2] + c[5]) * 0.5).copy())
                else:
                    cache.append(None)
            return cache
            
        # --- Helper for Isotropic Hand Scaling (External Pivot) ---
        def scale_hand_isotropically(hand_array, scale_factor, pivot):
            """
            Scales the hand points isotropically around an external pivot (Body Wrist).
            """
            matrix_ops_external.scale_about_root(hand_array, pivot, scale_factor, eps=0.01)

        # --- Helper for Isotropic Foot Scaling (External Pivot) ---
        def scale_foot_isotropically(foot_array, scale_factor, pivot):
            """
            Scales the foot points isotropically around an external pivot (Body Ankle).
            """
            matrix_ops_external.scale_about_root(foot_array, pivot, scale_factor, eps=0.01)

        def scale_foot_by_edges(foot_array, edge1_scale, edge2_scale, edge3_scale, body_to_foot_scale, pivot):
            """
            Scales foot internal shape by independent edge scaling.
            Step 1: Scale toes relative to current foot ankle (independent edges)
            Step 2: Enforce edge3 consistency between toes
            """
            if not isinstance(foot_array, np.ndarray) or foot_array.shape[0] < 3:
                return

            # Keep legacy internal foot-shape topology:
            # root=idx0, toe_a=idx1, toe_b=idx2.
            root = foot_array[0].copy()
            toe_a = foot_array[1].copy()
            toe_b = foot_array[2].copy()

            has_root = np.sum(np.abs(root)) > 0.01
            has_toe_a = np.sum(np.abs(toe_a)) > 0.01
            has_toe_b = np.sum(np.abs(toe_b)) > 0.01

            if not has_root:
                return

            # Step 1: Scale toes relative to current ankle position (independent edges)
            new_root = np.array([root[0], root[1]])

            if has_toe_a:
                new_toe_a_x = new_root[0] + (toe_a[0] - root[0]) * edge1_scale
                new_toe_a_y = new_root[1] + (toe_a[1] - root[1]) * edge1_scale
                if len(toe_a) >= 3:
                    foot_array[1] = [new_toe_a_x, new_toe_a_y, toe_a[2]]
                else:
                    foot_array[1] = [new_toe_a_x, new_toe_a_y]

            if has_toe_b:
                new_toe_b_x = new_root[0] + (toe_b[0] - root[0]) * edge2_scale
                new_toe_b_y = new_root[1] + (toe_b[1] - root[1]) * edge2_scale
                if len(toe_b) >= 3:
                    foot_array[2] = [new_toe_b_x, new_toe_b_y, toe_b[2]]
                else:
                    foot_array[2] = [new_toe_b_x, new_toe_b_y]

            # Step 2: Enforce edge3 (big toe <-> small toe) by symmetric correction around toe midpoint.
            if has_toe_a and has_toe_b:
                orig_edge3 = np.sqrt((toe_a[0] - toe_b[0]) ** 2 + (toe_a[1] - toe_b[1]) ** 2)
                if orig_edge3 > 1e-6:
                    target_edge3 = float(orig_edge3) * float(edge3_scale)
                    new_toe_a = np.array([float(foot_array[1][0]), float(foot_array[1][1])], dtype=float)
                    new_toe_b = np.array([float(foot_array[2][0]), float(foot_array[2][1])], dtype=float)
                    v = new_toe_b - new_toe_a
                    curr_len = float(np.sqrt(np.sum(v * v)))
                    # Degenerate toe direction: skip edge3 correction to avoid unstable direction.
                    if curr_len > 1e-6 and target_edge3 > 0.0:
                        toe_mid = (new_toe_a + new_toe_b) * 0.5
                        dir_vec = v / curr_len
                        half_len = 0.5 * target_edge3
                        corr_toe_a = toe_mid - dir_vec * half_len
                        corr_toe_b = toe_mid + dir_vec * half_len
                        foot_array[1][0] = float(corr_toe_a[0])
                        foot_array[1][1] = float(corr_toe_a[1])
                        foot_array[2][0] = float(corr_toe_b[0])
                        foot_array[2][1] = float(corr_toe_b[1])

        def apply_body_to_foot_placement_with_raw_direction(frame_data, raw_candidate, raw_feet, body_to_foot_scale):
            """
            Apply body-to-foot-ankle scaling at the end of scaling chain.
            Use raw per-frame body->foot vector directly and scale it by FK.
            """
            candidate = frame_data['bodies']['candidate']
            feet = frame_data['feet']
            side_cfg = [(0, 13), (1, 10)]  # (foot index, body ankle index)

            for foot_idx, body_idx in side_cfg:
                if foot_idx >= len(feet) or body_idx >= len(candidate):
                    continue
                foot = feet[foot_idx]
                if not isinstance(foot, np.ndarray) or foot.shape[0] < 1:
                    continue

                body_now = candidate[body_idx]
                # Heel (index 2) is used as foot root.
                foot_now = foot[2].copy()

                raw_body = raw_candidate[body_idx] if body_idx < len(raw_candidate) else None
                raw_foot = raw_feet[foot_idx][2] if foot_idx < len(raw_feet) and len(raw_feet[foot_idx]) > 2 else None

                if raw_body is None or raw_foot is None:
                    continue
                if not (has_pt(body_now) and has_pt(foot_now) and has_pt(raw_body) and has_pt(raw_foot)):
                    continue

                raw_vec = raw_foot - raw_body
                if float(np.sqrt(np.sum(raw_vec * raw_vec))) <= 1e-6:
                    continue
                foot_target = body_now + raw_vec * body_to_foot_scale
                delta = foot_target - foot_now

                for k in range(min(3, foot.shape[0])):
                    if has_pt(foot[k]):
                        foot[k] += delta

        # --- Helpers for Independent Extremity Lengths (Physical Pixel Coordinates) ---
        # Uses unified physical distance calculation to prevent coordinate system mismatch
        def get_max_hand_len(hand_pts):
            if not isinstance(hand_pts, np.ndarray) or hand_pts.shape[0] < 21: return 0.0
            if not has_pt(hand_pts[0]): return 0.0
            tips = [4, 8, 12, 16, 20]
            max_d = 0.0
            for t in tips:
                if has_pt(hand_pts[t]):
                    d = self._get_physical_dist(hand_pts[0], hand_pts[t])
                    if d > max_d: max_d = d
            return max_d

        def get_max_foot_len(foot_pts):
            if not isinstance(foot_pts, np.ndarray) or foot_pts.shape[0] < 3: return 0.0
            pts = []
            for i in range(3):
                if has_pt(foot_pts[i]): pts.append(foot_pts[i])
            if len(pts) < 2: return 0.0
            max_d = 0.0
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    d = self._get_physical_dist(pts[i], pts[j])
                    if d > max_d: max_d = d
            return max_d

        def get_arm_len(c):
            len_r = 0.0
            if has_pt(c[2]) and has_pt(c[3]): len_r += self._get_physical_dist(c[2], c[3])
            if has_pt(c[3]) and has_pt(c[4]): len_r += self._get_physical_dist(c[3], c[4])
            len_l = 0.0
            if has_pt(c[5]) and has_pt(c[6]): len_l += self._get_physical_dist(c[5], c[6])
            if has_pt(c[6]) and has_pt(c[7]): len_l += self._get_physical_dist(c[6], c[7])
            return max(len_r, len_l)

        def get_leg_len(c):
            len_r = 0.0
            if has_pt(c[8]) and has_pt(c[9]): len_r += self._get_physical_dist(c[8], c[9])
            if has_pt(c[9]) and has_pt(c[10]): len_r += self._get_physical_dist(c[9], c[10])
            len_l = 0.0
            if has_pt(c[11]) and has_pt(c[12]): len_l += self._get_physical_dist(c[11], c[12])
            if has_pt(c[12]) and has_pt(c[13]): len_l += self._get_physical_dist(c[12], c[13])
            return max(len_r, len_l)

        def clone_and_conf_gate_pose_data(pose_data):
            candidate = pose_data['bodies']['candidate'].copy()
            candidate_conf = pose_data['bodies']['candidate_conf'].copy()
            faces = pose_data['faces'].copy()
            faces_conf = pose_data['faces_conf'].copy()
            hands = pose_data['hands'].copy()
            hands_conf = pose_data['hands_conf'].copy()
            feet = pose_data['feet'].copy()
            feet_conf = pose_data['feet_conf'].copy()

            candidate[candidate_conf < conf_thresh] = 0.0
            if len(faces) > 0 and len(faces_conf) > 0:
                faces[0][faces_conf[0] < conf_thresh] = 0.0
            for side in range(min(len(hands), len(hands_conf))):
                hands[side][hands_conf[side] < conf_thresh] = 0.0
            for side in range(min(len(feet), len(feet_conf))):
                feet[side][feet_conf[side] < conf_thresh] = 0.0

            return candidate, candidate_conf, faces, faces_conf, hands, hands_conf, feet, feet_conf

        def validate_manual_anchor_override(manual_pose_data):
            """Validate manual anchor input using the same confidence gating strategy."""
            if manual_pose_data is None:
                return False, None, None, None, None, None

            c_m, c_m_conf, f_m, _, h_m, _, feet_m, _ = clone_and_conf_gate_pose_data(manual_pose_data)
            is_valid = has_pt(c_m[1]) and has_pt(c_m[2]) and has_pt(c_m[5]) and has_pt(c_m[8]) and has_pt(c_m[11])
            if is_valid:
                print("[WSCS] Manual Anchor Pose detected and validated. Bypassing Auto WSCS & Z-Axis Filter.")
            else:
                print("[WSCS] WARNING: Manual Anchor Pose is missing critical points (Neck, Shoulders, or Hips). Ignoring manual input and falling back to Auto WSCS.")
            return is_valid, c_m, c_m_conf, f_m, h_m, feet_m

        def resolve_anchor_aux_data(use_manual):
            """Resolve anchor conf/face/hand/feet data from manual or auto source."""
            if use_manual:
                return manual_candidate_conf, manual_faces, manual_hands, manual_feet
            return auto_candidate_conf, auto_faces, auto_hands, auto_feet

        def resolve_auto_anchor_data(selected_anchor_idx):
            """Resolve auto-anchor pose tensors from a selected anchor index."""
            anchor_pose_local = batch_pose_data[selected_anchor_idx]
            return clone_and_conf_gate_pose_data(anchor_pose_local)

        def initialize_anchor_state(manual_pose_data):
            """Initialize manual/auto anchor state with manual override validation."""
            state = {
                'is_manual_anchor_valid': False,
                'manual_candidate': None,
                'manual_candidate_conf': None,
                'manual_faces': None,
                'manual_hands': None,
                'manual_feet': None,
                'auto_candidate_conf': None,
                'auto_faces': None,
                'auto_hands': None,
                'auto_feet': None,
            }
            (state['is_manual_anchor_valid'],
             state['manual_candidate'],
             state['manual_candidate_conf'],
             state['manual_faces'],
             state['manual_hands'],
             state['manual_feet']) = validate_manual_anchor_override(manual_pose_data)
            return state

        # Source data: conf<threshold treated as non-existent.
        # Left-right complement is handled as "length complement" inside FK extractors.
        ref_candidate, ref_candidate_conf, ref_faces, ref_faces_conf, ref_hands, ref_hands_conf, ref_feet, ref_feet_conf = clone_and_conf_gate_pose_data(ref_data)
        
        # --- Manual Anchor Override Validation (same strategy as reference gating) ---
        anchor_state = initialize_anchor_state(manual_ref_data)
        is_manual_anchor_valid = anchor_state['is_manual_anchor_valid']
        manual_candidate = anchor_state['manual_candidate']
        manual_candidate_conf = anchor_state['manual_candidate_conf']
        manual_faces = anchor_state['manual_faces']
        manual_hands = anchor_state['manual_hands']
        manual_feet = anchor_state['manual_feet']
        auto_candidate_conf = anchor_state['auto_candidate_conf']
        auto_faces = anchor_state['auto_faces']
        auto_hands = anchor_state['auto_hands']
        auto_feet = anchor_state['auto_feet']
        if is_manual_anchor_valid:
            anchor_candidate = manual_candidate.copy()
            # Manual anchor is a strict override: disable auto best-hand / best-neck FK modifiers.
            best_hand_search = False
            best_neck_search = False
        
        anchor_idx = 0 # Initialize anchor_idx for safe return
        # Only run auto WSCS if manual override is not present or invalid.
        if not is_manual_anchor_valid:
            anchor_idx, _, _, _, _, _ = select_wscs_anchor(
                batch_pose_data=batch_pose_data,
                conf_thresh=conf_thresh,
                has_pt=has_pt,
                get_dist=get_dist,
                logger=print,
            )
            anchor_candidate, auto_candidate_conf, auto_faces, _, auto_hands, _, auto_feet, _ = resolve_auto_anchor_data(anchor_idx)
        # ========== Scale Constants ==========
        # Extract anchor frame data
        anchor_candidate_conf, anchor_faces, anchor_hands, anchor_feet = resolve_anchor_aux_data(is_manual_anchor_valid)

        # Shared hand length metric (middle finger chain 0-9-10-11-12).
        def get_mid_finger_chain_len(hand_pts):
            if not isinstance(hand_pts, np.ndarray) or hand_pts.shape[0] < 13:
                return None
            segs = [(0, 9), (9, 10), (10, 11), (11, 12)]
            total = 0.0
            for a, b in segs:
                if not (has_pt(hand_pts[a]) and has_pt(hand_pts[b])):
                    return None
                total += get_dist(hand_pts[a], hand_pts[b])
            return total if total > 1e-6 else None

        # Find best hand baseline:
        # 1) frame shoulder length within 5% of anchor shoulder
        # 2) either left or right arm's upper+lower lengths both within 5% of anchor side
        # 3) collect that side hand topology length (0-9-10-11-12), use the maximum as baseline
        def find_best_hand_baseline(anchor_c, frames):
            def rel_diff(a, b):
                if a is None or b is None or b <= 1e-6:
                    return None
                return abs(a - b) / b

            def hand_all_conf_ge(hand_conf_arr, thresh):
                if not isinstance(hand_conf_arr, np.ndarray) or hand_conf_arr.shape[0] < 21:
                    return False
                return bool(np.all(hand_conf_arr[:21] >= thresh))

            def is_mid_chain_outlier(hand_pts):
                if not isinstance(hand_pts, np.ndarray) or hand_pts.shape[0] < 13:
                    return True
                segs = [(0, 9), (9, 10), (10, 11), (11, 12)]
                for a, b in segs:
                    if not (has_pt(hand_pts[a]) and has_pt(hand_pts[b])):
                        return True
                    seg_len = get_dist(hand_pts[a], hand_pts[b])
                    if seg_len is None or seg_len <= 1e-6:
                        return True
                return False

            def has_finger_segment_longer_than_forearm(hand_pts, forearm_len):
                if not isinstance(hand_pts, np.ndarray) or hand_pts.shape[0] < 21:
                    return True
                if forearm_len is None or forearm_len <= 1e-6:
                    return True
                finger_segs = [
                    (1, 2), (2, 3), (3, 4),
                    (5, 6), (6, 7), (7, 8),
                    (9, 10), (10, 11), (11, 12),
                    (13, 14), (14, 15), (15, 16),
                    (17, 18), (18, 19), (19, 20),
                ]
                for a, b in finger_segs:
                    if not (has_pt(hand_pts[a]) and has_pt(hand_pts[b])):
                        continue
                    seg_len = get_dist(hand_pts[a], hand_pts[b])
                    if seg_len is not None and seg_len > forearm_len:
                        return True
                return False

            def len_safe(c, i, j):
                if has_pt(c[i]) and has_pt(c[j]):
                    return get_dist(c[i], c[j])
                return None

            anchor_shoulder_len = len_safe(anchor_c, 2, 5)
            anchor_upper_r = len_safe(anchor_c, 2, 3)
            anchor_lower_r = len_safe(anchor_c, 3, 4)
            anchor_upper_l = len_safe(anchor_c, 5, 6)
            anchor_lower_l = len_safe(anchor_c, 6, 7)

            hand_candidates = []
            if anchor_shoulder_len is not None:
                for fi, frame in enumerate(frames):
                    c = frame['bodies']['candidate']
                    hands = frame['hands']
                    hands_conf = frame.get('hands_conf')
                    sh_len = len_safe(c, 2, 5)
                    d_sh = rel_diff(sh_len, anchor_shoulder_len)
                    if d_sh is None or d_sh > 0.05:
                        continue

                    # Right side match -> collect right hand (index 1)
                    up_r = len_safe(c, 2, 3)
                    low_r = len_safe(c, 3, 4)
                    d_up_r = rel_diff(up_r, anchor_upper_r)
                    d_low_r = rel_diff(low_r, anchor_lower_r)
                    if d_up_r is not None and d_low_r is not None and d_up_r <= 0.05 and d_low_r <= 0.05:
                        right_conf_ok = (
                            isinstance(hands_conf, np.ndarray)
                            and hands_conf.shape[0] > 1
                            and hand_all_conf_ge(hands_conf[1], conf_thresh)
                        )
                        if not right_conf_ok:
                            continue
                        hand_len_r = get_mid_finger_chain_len(hands[1]) if len(hands) > 1 else None
                        if hand_len_r is not None and is_mid_chain_outlier(hands[1]):
                            hand_len_r = None
                        if hand_len_r is not None and has_finger_segment_longer_than_forearm(hands[1], low_r):
                            hand_len_r = None
                        if hand_len_r is not None:
                            hand_candidates.append({
                                "length": hand_len_r,
                                "side": "right",
                                "frame_index": fi,
                                "pts": hands[1].copy(),
                                "conf": hands_conf[1].copy() if isinstance(hands_conf, np.ndarray) and len(hands_conf) > 1 else np.zeros((21,)),
                            })

                    # Left side match -> collect left hand (index 0)
                    up_l = len_safe(c, 5, 6)
                    low_l = len_safe(c, 6, 7)
                    d_up_l = rel_diff(up_l, anchor_upper_l)
                    d_low_l = rel_diff(low_l, anchor_lower_l)
                    if d_up_l is not None and d_low_l is not None and d_up_l <= 0.05 and d_low_l <= 0.05:
                        left_conf_ok = (
                            isinstance(hands_conf, np.ndarray)
                            and hands_conf.shape[0] > 0
                            and hand_all_conf_ge(hands_conf[0], conf_thresh)
                        )
                        if not left_conf_ok:
                            continue
                        hand_len_l = get_mid_finger_chain_len(hands[0]) if len(hands) > 0 else None
                        if hand_len_l is not None and is_mid_chain_outlier(hands[0]):
                            hand_len_l = None
                        if hand_len_l is not None and has_finger_segment_longer_than_forearm(hands[0], low_l):
                            hand_len_l = None
                        if hand_len_l is not None:
                            hand_candidates.append({
                                "length": hand_len_l,
                                "side": "left",
                                "frame_index": fi,
                                "pts": hands[0].copy(),
                                "conf": hands_conf[0].copy() if isinstance(hands_conf, np.ndarray) and len(hands_conf) > 0 else np.zeros((21,)),
                            })

            if len(hand_candidates) == 0:
                return None, None
            best = max(hand_candidates, key=lambda x: x["length"])
            return best["length"], best

        def build_avg_hand_from_anchor_hands(anc_hands):
            if not isinstance(anc_hands, np.ndarray) or anc_hands.shape[0] < 2:
                return None, None
            left = anc_hands[0]
            right = anc_hands[1]
            if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
                return None, None

            n = min(left.shape[0], right.shape[0])
            avg_pts = np.zeros((n, 2))
            avg_conf = np.zeros((n,))
            for i in range(n):
                lv = has_pt(left[i])
                rv = has_pt(right[i])
                if lv and rv:
                    avg_pts[i] = (left[i] + right[i]) * 0.5
                    avg_conf[i] = 1.0
                elif lv:
                    avg_pts[i] = left[i].copy()
                    avg_conf[i] = 1.0
                elif rv:
                    avg_pts[i] = right[i].copy()
                    avg_conf[i] = 1.0

            avg_len = get_mid_finger_chain_len(avg_pts)
            if avg_len is None:
                return None, None
            return avg_len, {
                "length": avg_len,
                "side": "both_avg",
                "frame_index": anchor_idx,
                "pts": avg_pts.copy(),
                "conf": avg_conf.copy(),
            }

        def find_best_neck_points(anchor_c, frames):
            def len_safe(c, i, j):
                if has_pt(c[i]) and has_pt(c[j]):
                    return get_dist(c[i], c[j])
                return None

            anchor_shoulder_len = len_safe(anchor_c, 2, 5)
            if anchor_shoulder_len is None or anchor_shoulder_len <= 1e-6:
                return None

            best = None
            for fi, frame in enumerate(frames):
                c = frame['bodies']['candidate']
                conf = frame['bodies']['candidate_conf']
                if len(conf) <= 5:
                    continue
                if not (has_pt(c[2]) and has_pt(c[5]) and has_pt(c[1]) and has_pt(c[0])):
                    continue
                if conf[2] < conf_thresh or conf[5] < conf_thresh or conf[1] < conf_thresh or conf[0] < conf_thresh:
                    continue

                sh_len = len_safe(c, 2, 5)
                if sh_len is None or sh_len <= 1e-6:
                    continue
                if abs(sh_len - anchor_shoulder_len) / anchor_shoulder_len > 0.05:
                    continue

                neck_len = len_safe(c, 0, 1)
                if neck_len is None or neck_len <= 1e-6:
                    continue

                if best is None or neck_len > best["length"]:
                    best = {
                        "length": float(neck_len),
                        "frame_index": fi,
                        "nose": c[0].copy(),
                        "neck": c[1].copy(),
                    }
            return best

        def build_fk_values(anc_candidate, anc_faces, anc_hands, anc_feet, hand_baseline):
            """Build FK package from reference and anchor data with existing protections."""
            def point_dist_if_valid(p1, p2):
                if has_pt(p1) and has_pt(p2):
                    return float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))
                return None

            fk_torso, fk_neck, fk_shoulder, fk_hip_width = self.extract_fk_values_part1_torso_neck(ref_candidate, anc_candidate)
            fk_upper_arm, fk_lower_arm, arm_ref_lock_to_long = self.extract_fk_values_part2_arms(ref_candidate, anc_candidate)
            fk_upper_leg, fk_lower_leg = self.extract_fk_values_part3_legs(ref_candidate, anc_candidate)
            part4_values = self.extract_fk_values_part4_hands_feet_face(
                ref_candidate, ref_faces, ref_hands, ref_feet,
                anc_candidate, anc_faces, anc_hands, anc_feet, hand_baseline
            )
            if len(part4_values) == 8:
                fk_hand, fk_foot_edge1, fk_foot_edge2, fk_foot_edge3, fk_body_to_foot_ankle, fk_face_x, fk_face_y, fk_eye_width = part4_values
            else:
                raise ValueError(f"Unexpected part4 return length: {len(part4_values)}")
            ref_eye_dist = point_dist_if_valid(ref_candidate[14], ref_candidate[15])
            anc_eye_dist = point_dist_if_valid(anc_candidate[14], anc_candidate[15])
            print(
                f"[Eye FK Debug] ref_eye={ref_eye_dist if ref_eye_dist is not None else 'None'}, "
                f"anc_eye={anc_eye_dist if anc_eye_dist is not None else 'None'}, "
                f"fk_eye_width={fk_eye_width:.6f}"
            )

            # Phase 2.5: Hand FK Anomaly Protection
            fk_hand = self.validate_hand_fk(fk_hand, fk_torso, fk_neck, fk_upper_arm, fk_lower_arm,
                                            fk_upper_leg, fk_lower_leg, fk_foot_edge1, fk_foot_edge2, fk_foot_edge3, fk_body_to_foot_ankle, fk_face_x, fk_face_y, fk_eye_width)

            # Optional mode: replace hand FK with shoulder FK.
            if use_shoulder_fk_for_hand:
                print(f"[Hand FK Mode] Shoulder-driven hand FK enabled: hand_fk <- shoulder_fk ({fk_shoulder:.3f})")
                fk_hand = fk_shoulder

            # Phase 2.6: Hand Scaling Switch
            if not hand_scaling:
                print("[Hand Scaling] Disabled by user. Setting hand FK to 1.0")
                fk_hand = 1.0

            return {
                'torso': fk_torso,
                'neck': fk_neck,
                'shoulder': fk_shoulder,
                'hip_width': fk_hip_width,
                'upper_arm': fk_upper_arm,
                'lower_arm': fk_lower_arm,
                'upper_leg': fk_upper_leg,
                'lower_leg': fk_lower_leg,
                'hand': fk_hand,
                'foot_edge1': fk_foot_edge1,
                'foot_edge2': fk_foot_edge2,
                'foot_edge3': fk_foot_edge3,
                'body_to_foot_ankle': fk_body_to_foot_ankle,
                'face_x': fk_face_x,
                'face_y': fk_face_y,
                'eye_width': fk_eye_width
            }

        def build_scale_package(anc_candidate, anc_faces, anc_hands, anc_feet, hand_baseline):
            """Build Global RPCA + FK package + final scales with unchanged execution order."""
            # Extract F0 (first frame) data
            f0_candidate = results_vis[0]['bodies']['candidate']
            f0_faces = results_vis[0]['faces']

            # Phase 1: Calculate 13-bone Global RPCA (Anchor vs F0)
            global_rpca = self.calculate_13_bone_global_rpca(
                anc_candidate, anc_faces,
                f0_candidate, f0_faces,
                alignment_mode
            )

            # Phase 2: Extract and assemble FK values (Ref vs Anchor)
            fk_pkg = build_fk_values(anc_candidate, anc_faces, anc_hands, anc_feet, hand_baseline)

            # Phase 3: Forge final scale constants (FK × RPCA)
            final_scale_pkg = self.forge_final_scale_constants(fk_pkg, global_rpca, alignment_mode)

            print(f"[New Pipeline] Global RPCA: {global_rpca:.3f}")
            print(f"[New Pipeline] Final Scales: torso={final_scale_pkg['torso']:.3f}, neck={final_scale_pkg['neck']:.3f}, "
                  f"shoulder={final_scale_pkg['shoulder']:.3f}, "
                  f"hip={final_scale_pkg['hip_width']:.3f}, "
                  f"arm={final_scale_pkg['upper_arm']:.3f}/{final_scale_pkg['lower_arm']:.3f}, "
                  f"leg={final_scale_pkg['upper_leg']:.3f}/{final_scale_pkg['lower_leg']:.3f}, "
                  f"hand={final_scale_pkg['hand']:.3f}, foot={final_scale_pkg['foot_edge1']:.3f}/{final_scale_pkg['foot_edge2']:.3f}/{final_scale_pkg['foot_edge3']:.3f}, "
                  f"body_to_foot_ankle={final_scale_pkg['body_to_foot_ankle']:.3f}, "
                  f"face={final_scale_pkg['face_x']:.3f}/{final_scale_pkg['face_y']:.3f}, eye={final_scale_pkg['eye_width']:.3f}")
            eye_adjust_ratio = (final_scale_pkg['eye_width'] / final_scale_pkg['face_x']) if abs(final_scale_pkg['face_x']) > 1e-6 else 1.0
            print(f"[Eye FK Debug] final_eye_adjust_ratio(eye/face_x)={eye_adjust_ratio:.6f}")

            return global_rpca, fk_pkg, final_scale_pkg

        def map_final_scales_to_compat_vars(scale_pkg):
            """Map final scales dict to per-frame compatibility variables."""
            return (
                scale_pkg['torso'],
                scale_pkg['neck'],
                scale_pkg['shoulder'],
                scale_pkg['hip_width'],
                scale_pkg['upper_arm'],
                scale_pkg['lower_arm'],
                scale_pkg['upper_leg'],
                scale_pkg['lower_leg'],
                scale_pkg['hand'],
                scale_pkg['foot_edge1'],
                scale_pkg['foot_edge2'],
                scale_pkg['foot_edge3'],
                scale_pkg['body_to_foot_ankle']
            )

        def extract_face_scales(scale_pkg):
            """Extract face X/Y scales used by downstream per-frame logic."""
            return scale_pkg['face_x'], scale_pkg['face_y'], scale_pkg['eye_width']

        def build_ref_lengths(ref_cand):
            """Build reference bone lengths used by RPCA pre-scan setup."""
            def get_ref_len(idx1, idx2):
                if has_pt(ref_cand[idx1]) and has_pt(ref_cand[idx2]):
                    return get_dist(ref_cand[idx1], ref_cand[idx2])
                return 0.0

            lengths = {
                'neck': get_ref_len(0, 1), # Nose-Neck usually, but using Neck logic below
                'neck_alt': get_ref_len(1, 0), # Fallback
                'shoulder_r': get_ref_len(1, 2), 'shoulder_l': get_ref_len(1, 5),
                'arm_up_r': get_ref_len(2, 3), 'arm_low_r': get_ref_len(3, 4),
                'arm_up_l': get_ref_len(5, 6), 'arm_low_l': get_ref_len(6, 7),
                'leg_up_r': get_ref_len(8, 9), 'leg_low_r': get_ref_len(9, 10),
                'leg_up_l': get_ref_len(11, 12), 'leg_low_l': get_ref_len(12, 13),
                # Head keys are kept to preserve downstream dictionary access.
                'head14': get_ref_len(0, 14), 'head15': get_ref_len(0, 15),
                'head16': get_ref_len(14, 16), 'head17': get_ref_len(15, 17)
            }
            # Special case for Neck in existing logic uses index 0 and 1
            lengths['neck_actual'] = get_ref_len(0, 1)

            # Calculate Torso (Spine) length for RPCA from Neck(1) to Mid-Hip
            # Note: existing logic used 1 and 8/11.
            if has_pt(ref_cand[1]) and has_pt(ref_cand[8]) and has_pt(ref_cand[11]):
                m_h_ref = (ref_cand[8] + ref_cand[11]) * 0.5
                lengths['torso'] = get_dist(ref_cand[1], m_h_ref)
            else:
                lengths['torso'] = 0.0
            return lengths

        def build_anchor_scale_metrics(anc_candidate):
            """Build anchor shoulder width and torso length for setup calculations."""
            anchor_shoulder_width = get_dist(anc_candidate[2], anc_candidate[5]) if (has_pt(anc_candidate[2]) and has_pt(anc_candidate[5])) else 1.0
            anchor_torso_length = 1.0
            if has_pt(anc_candidate[8]) and has_pt(anc_candidate[11]) and has_pt(anc_candidate[2]) and has_pt(anc_candidate[5]):
                m_s = (anc_candidate[2] + anc_candidate[5]) * 0.5
                m_h = (anc_candidate[8] + anc_candidate[11]) * 0.5
                anchor_torso_length = get_dist(m_s, m_h)
            return anchor_shoulder_width, anchor_torso_length

        def build_torso_validity_distances(frames):
            """Build torso validity list and forward/backward distances to valid torso frames."""
            has_torso = []
            for frame in frames:
                c = frame['bodies']['candidate']
                # Condition: Neck(1) and Hips(8,11) for Mid_Hip
                has_t = has_pt(c[1]) and has_pt(c[8]) and has_pt(c[11])
                has_torso.append(has_t)

            total = len(frames)
            d_in = [float('inf')] * total
            d_out = [float('inf')] * total

            # Pass 1: d_in (distance to previous valid)
            last_valid_idx = -1
            for i in range(total):
                if has_torso[i]:
                    last_valid_idx = i
                elif last_valid_idx != -1:
                    d_in[i] = i - last_valid_idx

            # Pass 2: d_out (distance to next valid)
            last_valid_idx = -1
            for i in range(total - 1, -1, -1):
                if has_torso[i]:
                    last_valid_idx = i
                elif last_valid_idx != -1:
                    d_out[i] = last_valid_idx - i

            return has_torso, d_in, d_out

        if best_hand_search:
            best_hand_baseline, best_hand = find_best_hand_baseline(anchor_candidate, batch_pose_data)
        else:
            best_hand_baseline, best_hand = build_avg_hand_from_anchor_hands(anchor_hands)

        # Optional: use best-neck candidate as FK anchor standard (does not change main transform anchor).
        fk_anchor_candidate = anchor_candidate
        if best_neck_search:
            best_neck = find_best_neck_points(anchor_candidate, batch_pose_data)
            if best_neck is not None:
                fk_anchor_candidate = anchor_candidate.copy()
                best_neck_vec = best_neck["nose"] - best_neck["neck"]
                fk_anchor_candidate[1] = anchor_candidate[1].copy()
                fk_anchor_candidate[0] = anchor_candidate[1].copy() + best_neck_vec
                print(f"[BestNeck FK] selected frame={best_neck['frame_index']}, neck_len={best_neck['length']:.4f}")

        global_rpca_multiplier, fk_values, final_scales = build_scale_package(
            fk_anchor_candidate, anchor_faces, anchor_hands, anchor_feet, best_hand_baseline
        )

        # --- [START RPCA Pre-scan & Setup] ---
        # 1. Pre-calculate Source Lengths (Constant targets)
        ref_lens = build_ref_lengths(ref_candidate)

        # 2. Global Anchor Data for Scale_Z
        anchor_w_shoulder, anchor_torso_len = build_anchor_scale_metrics(anchor_candidate)

        # ========== Phase 4: Fixed Scale Application ==========
        # Use fixed FK × RPCA values.
        # Note: The new system already calculated these in physical pixel domain
        # and they already include the global RPCA multiplier

        # Map final_scales to compatibility variables used by the per-frame loop.
        true_body_scale, true_neck_scale, true_shoulder_scale, true_hip_width_scale, \
        true_upper_arm_scale, true_lower_arm_scale, true_upper_leg_scale, true_lower_leg_scale, \
        true_hand_scale, true_foot_edge1_scale, true_foot_edge2_scale, true_foot_edge3_scale, \
        true_body_to_foot_ankle_scale = map_final_scales_to_compat_vars(final_scales)

        # Note: Face scales are handled separately in the per-frame loop
        # They use final_scales['face_x'] and final_scales['face_y']

        print(f"[Phase 4] Using new fixed scales: body={true_body_scale:.3f}, neck={true_neck_scale:.3f}, "
              f"shoulder={true_shoulder_scale:.3f}, "
              f"hip={true_hip_width_scale:.3f}, "
              f"upper_arm={true_upper_arm_scale:.3f}, lower_arm={true_lower_arm_scale:.3f}, "
              f"upper_leg={true_upper_leg_scale:.3f}, lower_leg={true_lower_leg_scale:.3f}, "
              f"hand={true_hand_scale:.3f}, foot={true_foot_edge1_scale:.3f}/{true_foot_edge2_scale:.3f}/{true_foot_edge3_scale:.3f}, "
              f"body_to_foot_ankle={true_body_to_foot_ankle_scale:.3f}")

        # Scale RPCA Source Lengths to enforce absolute pixel alignment for Torso/Spine
        for k in ref_lens:
            ref_lens[k] *= 1.0  # Note: RPCA multiplier already included in true_*_scale values
        # --- [END] Phase 4 ---

        # ========== Use New Face Scales ==========
        # The new system calculates face scales independently from shoulders
        # Use the new final_scales values directly
        final_face_x, final_face_y, final_eye_width = extract_face_scales(final_scales)

        print(f"[Phase 4] Using new face scales: face_x={final_face_x:.3f}, face_y={final_face_y:.3f}, eye={final_eye_width:.3f}")

        # 3. Build P1 Validity List (Has Torso)
        has_torso_list, d_in_list, d_out_list = build_torso_validity_distances(batch_pose_data)


        # --- [START Spine Offset Pre-calculation Pipeline] ---
        # Use fixed FK value from final_scales instead of dynamic scale_z
        torso_scale = final_scales['torso']
        neck_scale = final_scales['neck']
        shoulder_scale = final_scales['shoulder']

        def precompute_core_offsets():
            """Precompute spine/neck/shoulder offsets with unchanged formulas and data sources."""
            batch_geom_cache = build_frame_geom_cache(batch_pose_data)
            vis_geom_cache = batch_geom_cache if results_vis is batch_pose_data else build_frame_geom_cache(results_vis)

            def build_offset_from_points(scale, primary_from, primary_to, fallback_from=None, fallback_to=None):
                if primary_from is not None and primary_to is not None:
                    return (primary_from - primary_to) * (1.0 - scale)
                if fallback_from is not None and fallback_to is not None:
                    return (fallback_from - fallback_to) * (1.0 - scale)
                return None

            raw_spine_offsets = [
                build_offset_from_points(
                    torso_scale,
                    e["neck"],
                    e["mid_hip"],
                    e["mid_shoulder"],
                    e["mid_hip"],
                )
                for e in batch_geom_cache
            ]
            raw_neck_offsets = [
                build_offset_from_points(
                    neck_scale,
                    e["neck"],
                    e["nose"],
                    e["mid_shoulder"],
                    e["nose"],
                )
                for e in vis_geom_cache
            ]
            raw_rshoulder_offsets = [
                build_offset_from_points(
                    shoulder_scale,
                    e["neck"],
                    e["r_shoulder"],
                    e["mid_shoulder"],
                    e["r_shoulder"],
                )
                for e in vis_geom_cache
            ]
            raw_lshoulder_offsets = [
                build_offset_from_points(
                    shoulder_scale,
                    e["neck"],
                    e["l_shoulder"],
                    e["mid_shoulder"],
                    e["l_shoulder"],
                )
                for e in vis_geom_cache
            ]

            spine_offsets = apply_bidirectional_interpolation(raw_spine_offsets)
            neck_offsets = apply_bidirectional_interpolation(raw_neck_offsets)
            rshoulder_offsets = apply_bidirectional_interpolation(raw_rshoulder_offsets)
            lshoulder_offsets = apply_bidirectional_interpolation(raw_lshoulder_offsets)
            return spine_offsets, neck_offsets, rshoulder_offsets, lshoulder_offsets

        # Time-offset setup completed.

        # 1. Select unified offset/stabilizer reference base
        base_offset_candidate = anchor_candidate
        base_offset_candidate_conf = anchor_candidate_conf
        base_offset_label = "anchor"
        if first_frame_offset_alignment and len(results_vis) > 0:
            selected_idx = -1
            for idx, frame in enumerate(results_vis):
                cand = frame['bodies']['candidate']
                conf = frame['bodies']['candidate_conf']
                neck_ok = has_pt(cand[1]) and (idx < len(results_vis)) and (len(conf) > 1 and conf[1] >= conf_thresh)
                shoulder_ok = has_pt(cand[2]) and has_pt(cand[5])
                if neck_ok or shoulder_ok:
                    selected_idx = idx
                    break

            if selected_idx >= 0:
                first_valid_candidate = results_vis[selected_idx]['bodies']['candidate']
                first_valid_conf = results_vis[selected_idx]['bodies']['candidate_conf']
                base_offset_candidate = first_valid_candidate
                base_offset_candidate_conf = first_valid_conf
                base_offset_label = "first_valid" if selected_idx > 0 else "first"

        def get_stabilizer_point(c, conf):
            def has_valid_pt(idx):
                return idx < len(conf) and has_pt(c[idx]) and conf[idx] >= conf_thresh
            # Priority 1: Neck
            if has_valid_pt(1):
                return c[1].copy()
            # Priority 2: Shoulder midpoint
            if has_valid_pt(2) and has_valid_pt(5):
                return ((c[2] + c[5]) * 0.5).copy()
            # Priority 3: Missing
            return None

        # 2. Calculate Global Base Offset using the same point policy as stabilizer.
        base_point = get_stabilizer_point(base_offset_candidate, base_offset_candidate_conf)
        ref_point = get_stabilizer_point(ref_candidate, ref_candidate_conf)
        global_base_offset = np.array([0.0, 0.0])
        if base_point is not None and ref_point is not None:
            global_base_offset = ref_point - base_point
        print(f"[Global Offset] base={base_offset_label}, first_frame_offset_alignment={first_frame_offset_alignment}")

        # --- Offset Stabilizer Constants (fixed over the whole video) ---
        def calc_body_metric(c):
            if not (has_pt(c[0]) and has_pt(c[2]) and has_pt(c[5])):
                return None
            shoulder_w = get_dist(c[2], c[5])
            if shoulder_w <= 1e-6:
                return None
            if has_pt(c[10]) and has_pt(c[13]):
                body_mid = (c[10] + c[13]) * 0.5
            elif has_pt(c[8]) and has_pt(c[11]):
                body_mid = (c[8] + c[11]) * 0.5
            else:
                return None
            body_len = get_dist(c[0], body_mid)
            if body_len <= 1e-6:
                return None
            return body_len / shoulder_w

        def build_stabilizer_components():
            """Build per-frame stabilizer compensation with unchanged ratio and fallback logic."""
            stabilizer_shoulder_ratio = (fk_values['shoulder'] * global_rpca_multiplier) if offset_stabilizer else 1.0
            stabilizer_body_ratio = 1.0
            body_metric_ref = calc_body_metric(ref_candidate)
            body_metric_base = calc_body_metric(base_offset_candidate)
            if offset_stabilizer and body_metric_ref is not None and body_metric_base is not None and body_metric_base > 1e-6:
                stabilizer_body_ratio = (body_metric_ref / body_metric_base) * global_rpca_multiplier

            anchor_stabilizer_point = get_stabilizer_point(base_offset_candidate, base_offset_candidate_conf)

            stabilizer_point_cache = build_stabilizer_point_cache(batch_pose_data)
            raw_stabilizer_comps = []
            for curr_point in stabilizer_point_cache:
                if not offset_stabilizer or anchor_stabilizer_point is None or curr_point is None:
                    raw_stabilizer_comps.append(None)
                    continue
                delta = curr_point - anchor_stabilizer_point
                raw_stabilizer_comps.append(np.array([
                    ((stabilizer_shoulder_ratio - 1.0) * delta[0]) if offset_stabilizer_x else 0.0,
                    (stabilizer_body_ratio - 1.0) * delta[1]
                ]))

            return apply_bidirectional_interpolation(raw_stabilizer_comps)

        def compute_time_offsets():
            """Compute all time-dependent offsets with unchanged interpolation and fallback rules."""
            spine_offsets, neck_offsets, rshoulder_offsets, lshoulder_offsets = precompute_core_offsets()
            stabilizer_comps = build_stabilizer_components()
            return spine_offsets, neck_offsets, rshoulder_offsets, lshoulder_offsets, stabilizer_comps

        final_spine_offsets, final_neck_offsets, final_rshoulder_offsets, final_lshoulder_offsets, final_stabilizer_comps = compute_time_offsets()

        def build_stabilized_offset(base_offset, frame_idx):
            return frame_ops_external.build_stabilized_offset(
                base_offset=base_offset,
                frame_idx=frame_idx,
                enable_final_offset_alignment=final_offset_alignment,
                enable_offset_stabilizer=offset_stabilizer,
                final_stabilizer_comps=final_stabilizer_comps,
            )

        def force_align_face_hands_to_body(frame_data):
            frame_ops_external.force_align_face_hands_to_body(frame_data, has_pt, safe_add)

        def apply_global_offset_to_frame(frame_data, offset):
            frame_ops_external.apply_global_offset_to_frame(frame_data, offset, has_pt, safe_add)

        def apply_neck_and_shoulder_offsets(frame_data, neck_offset, right_shoulder_offset, left_shoulder_offset):
            frame_ops_external.apply_neck_and_shoulder_offsets(
                frame_data, neck_offset, right_shoulder_offset, left_shoulder_offset, has_pt, safe_add
            )

        def apply_arm_chain_offsets(frame_data, raw_candidate, upper_arm_ratio, lower_arm_ratio):
            frame_ops_external.apply_arm_chain_offsets(frame_data, raw_candidate, upper_arm_ratio, lower_arm_ratio, has_pt, safe_add)

        def apply_leg_chain_offsets(frame_data, raw_candidate, hip_width_scale, ll1_ratio, ll2_ratio, rl1_ratio, rl2_ratio):
            frame_ops_external.apply_leg_chain_offsets(
                frame_data, raw_candidate, hip_width_scale, ll1_ratio, ll2_ratio, rl1_ratio, rl2_ratio, has_pt, safe_add
            )

        def apply_rigid_head_points(candidate, original_candidate, face_x_scale, face_y_scale, eye_width_scale):
            frame_ops_external.apply_rigid_head_points(candidate, original_candidate, face_x_scale, face_y_scale, eye_width_scale, has_pt)

        def apply_face_rigid_mask(faces, original_faces, face_x_scale, face_y_scale):
            frame_ops_external.apply_face_rigid_mask(faces, original_faces, face_x_scale, face_y_scale, safe_add)

        def try_apply_face_rigid_mask(frame_data, original_faces, face_x_scale, face_y_scale, require_nonzero_sum=False):
            frame_ops_external.try_apply_face_rigid_mask(
                frame_data, original_faces, face_x_scale, face_y_scale, has_pt, safe_add, require_nonzero_sum=require_nonzero_sum
            )

        def apply_face_mask_for_frame(frame_data, face_x_scale, face_y_scale, eye_width_scale, require_nonzero_sum=False):
            frame_ops_external.apply_face_mask_for_frame(
                frame_data, face_x_scale, face_y_scale, eye_width_scale, has_pt, safe_add, require_nonzero_sum=require_nonzero_sum
            )

        def get_limb_chain_ratios():
            """Get fixed limb chain ratios from precomputed final scales."""
            arm2_ratio = true_upper_arm_scale  # Right upper arm (shoulder to elbow)
            arm3_ratio = true_lower_arm_scale  # Right lower arm (elbow to wrist)
            ll1_ratio = true_upper_leg_scale  # Left thigh (hip to knee)
            ll2_ratio = true_lower_leg_scale  # Left calf (knee to ankle)
            rl1_ratio = true_upper_leg_scale  # Right thigh (hip to knee)
            rl2_ratio = true_lower_leg_scale  # Right calf (knee to ankle)
            return arm2_ratio, arm3_ratio, ll1_ratio, ll2_ratio, rl1_ratio, rl2_ratio

        def apply_spine_offset_to_lower_body(frame_data, spine_offset):
            frame_ops_external.apply_spine_offset_to_lower_body(frame_data, spine_offset, has_pt, safe_add)

        def apply_extremity_scaling(frame_data, candidate_points):
            frame_ops_external.apply_extremity_scaling(
                frame_data,
                candidate_points,
                true_hand_scale,
                true_foot_edge1_scale,
                true_foot_edge2_scale,
                true_foot_edge3_scale,
                true_body_to_foot_ankle_scale,
                foot_scaling,
                scale_hand_isotropically,
                scale_foot_by_edges,
            )

        def get_face_scale_pair():
            """Return fixed face scales for current frame."""
            return final_face_x, final_face_y, final_eye_width

        def prepare_frame_context(frame_data):
            """Prepare common frame context values used by both frame-0 and loop paths."""
            candidate_local = frame_data['bodies']['candidate']
            original_candidate_local = candidate_local.copy()
            face_x_local, face_y_local, eye_width_local = get_face_scale_pair()
            arm2_local, arm3_local, ll1_local, ll2_local, rl1_local, rl2_local = get_limb_chain_ratios()
            return (
                candidate_local,
                original_candidate_local,
                face_x_local,
                face_y_local,
                eye_width_local,
                arm2_local,
                arm3_local,
                ll1_local,
                ll2_local,
                rl1_local,
                rl2_local
            )

        def apply_frame_body_pipeline(frame_data, frame_idx, original_candidate, raw_candidate, raw_feet, face_x_scale, face_y_scale, eye_width_scale, arm2_ratio, arm3_ratio, ll1_ratio, ll2_ratio, rl1_ratio, rl2_ratio):
            """Apply the full per-frame body transform chain with unchanged execution order."""
            apply_neck_and_shoulder_offsets(
                frame_data,
                final_neck_offsets[frame_idx],
                final_rshoulder_offsets[frame_idx],
                final_lshoulder_offsets[frame_idx]
            )
            apply_arm_chain_offsets(frame_data, raw_candidate, arm2_ratio, arm3_ratio)

            candidate_local = frame_data['bodies']['candidate']
            apply_rigid_head_points(candidate_local, original_candidate, face_x_scale, face_y_scale, eye_width_scale)

            offset_spine = final_spine_offsets[frame_idx]
            apply_spine_offset_to_lower_body(frame_data, offset_spine)

            apply_leg_chain_offsets(
                frame_data,
                raw_candidate,
                true_hip_width_scale,
                ll1_ratio,
                ll2_ratio,
                rl1_ratio,
                rl2_ratio
            )

            # Final foot placement for scaling chain:
            # use raw direction and apply body_to_foot_ankle after body scaling.
            apply_body_to_foot_placement_with_raw_direction(
                frame_data,
                raw_candidate,
                raw_feet,
                true_body_to_foot_ankle_scale
            )

            offset = build_stabilized_offset(global_base_offset, frame_idx)
            apply_global_offset_to_frame(frame_data, offset)
            force_align_face_hands_to_body(frame_data)

        def run_frame_pipeline(frame_data, frame_idx, require_nonzero_sum, apply_extremity_before_scale):
            """Run frame preprocessing + transform pipeline with ordering controls."""
            if apply_extremity_before_scale:
                apply_extremity_scaling(frame_data, frame_data['bodies']['candidate'])

            raw_candidate = frame_data['bodies']['candidate'].copy()
            raw_feet = frame_data['feet'].copy()

            candidate_local, original_candidate, face_x_scale, face_y_scale, eye_width_scale, arm2_ratio, arm3_ratio, ll1_ratio, ll2_ratio, rl1_ratio, rl2_ratio = prepare_frame_context(frame_data)
            apply_face_mask_for_frame(frame_data, face_x_scale, face_y_scale, eye_width_scale, require_nonzero_sum=require_nonzero_sum)

            if not apply_extremity_before_scale:
                apply_extremity_scaling(frame_data, candidate_local)

            apply_frame_body_pipeline(
                frame_data, frame_idx, original_candidate, raw_candidate, raw_feet, face_x_scale, face_y_scale, eye_width_scale,
                arm2_ratio, arm3_ratio, ll1_ratio, ll2_ratio, rl1_ratio, rl2_ratio
            )

        # Frame 0 transform application.
        run_frame_pipeline(
            results_vis[0], 0,
            require_nonzero_sum=True,
            apply_extremity_before_scale=False
        )
        
        # --- Batch Propagation ---
        for i in range(1, len(results_vis)):
            run_frame_pipeline(
                results_vis[i], i,
                require_nonzero_sum=False,
                apply_extremity_before_scale=False
            )

        # ========== Phase 5: Convert Back to Normalized Coordinates ==========
        if not output_absolute_coordinates:
            self.convert_to_normalized_coords(results_vis, canvas_width, canvas_height)

        # Return transformed frames and selected anchor index.
        return results_vis, anchor_idx, best_hand


# Backward/forward-compatible alias.




