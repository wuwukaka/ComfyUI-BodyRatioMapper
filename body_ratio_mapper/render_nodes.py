"""
BodyRatioMapper render node classes
Contains nodes for rendering POSE_KEYPOINT data using SDPose styling
"""

import numpy as np
import torch
import cv2
import math
import matplotlib.colors


# ==============================================================================
#  SDPose (WholeBody) Render Node - Strictly Aligned with SDPose OOD
# ==============================================================================

def draw_sdpose_wholebody_standard(
    canvas,
    keypoints,
    scores=None,
    threshold=0.3,
    scale_for_xinsir=False,
    stick_width=4,
    face_point_size=3,
    draw_face=True,
    draw_mouth=True,
    draw_hands=True,
    draw_feet=True,
):
    """
    Render whole-body keypoints in SDPose/OpenPose style.
    """
    H, W, C = canvas.shape
    
    # Base stroke width configuration.
    base_stickwidth = max(1, int(stick_width))
    stickwidth = base_stickwidth
    face_radius = max(1, int(face_point_size))

    # Match Xinsir adaptive thickness scaling.
    if scale_for_xinsir:
        target_max_side = max(H, W)
        xinsir_stick_scale = 1 if target_max_side < 500 else min(2 + (target_max_side // 1000), 7)
        stickwidth = int(base_stickwidth * xinsir_stick_scale)

    body_limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]
    hand_edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    # 1. Draw body and feet.
    if len(keypoints) >= 18:
        for i, limb in enumerate(body_limbSeq):
            idx1, idx2 = limb[0] - 1, limb[1] - 1
            if idx1 >= 18 or idx2 >= 18: continue
            if scores is not None and (scores[idx1] < threshold or scores[idx2] < threshold): continue
            
            # Keep coordinate mapping aligned with OOD: index 0 = X, index 1 = Y.
            pt1, pt2 = keypoints[idx1], keypoints[idx2]
            mX, mY = (pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2
            length = np.sqrt(np.sum((pt1 - pt2)**2))
            
            if length < 1: continue
            angle = math.degrees(math.atan2(pt1[1] - pt2[1], pt1[0] - pt2[0]))
            polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i % len(colors)])
        
        for i in range(18):
            if scores is not None and scores[i] < threshold: continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= x < W and 0 <= y < H: 
                cv2.circle(canvas, (x, y), 4, colors[i % len(colors)], thickness=-1)

    if draw_feet and len(keypoints) >= 24:
        for i in range(18, 24):
            if scores is not None and scores[i] < threshold: continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= x < W and 0 <= y < H: 
                cv2.circle(canvas, (x, y), 4, colors[i % len(colors)], thickness=-1)

    # 2. Draw hands.
    def _draw_hands(start_idx):
        for ie, edge in enumerate(hand_edges):
            idx1, idx2 = start_idx + edge[0], start_idx + edge[1]
            if scores is not None and (scores[idx1] < threshold or scores[idx2] < threshold): continue
            x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1])
            x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])
            if x1 > 0.01 and y1 > 0.01 and x2 > 0.01 and y2 > 0.01 and 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                color = np.array(matplotlib.colors.hsv_to_rgb([ie / float(len(hand_edges)), 1.0, 1.0])) * 255
                cv2.line(canvas, (x1, y1), (x2, y2), color.astype(int).tolist(), thickness=stickwidth)
        for i in range(start_idx, start_idx + 21):
            if scores is not None and scores[i] < threshold: continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if x > 0.01 and y > 0.01 and 0 <= x < W and 0 <= y < H: 
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)

    if draw_hands:
        if len(keypoints) >= 113: _draw_hands(92) # Right hand
        if len(keypoints) >= 134: _draw_hands(113) # Left hand

    # 3. Draw face.
    if draw_face and len(keypoints) >= 92:
        for i in range(24, 92):
            # Face landmarks are mapped to global [24..91].
            # Mouth landmarks in 68-face indexing are [48..67], i.e. global [72..91].
            if not draw_mouth and i >= 72:
                continue
            if scores is not None and scores[i] < threshold: continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if x > 0.01 and y > 0.01 and 0 <= x < W and 0 <= y < H: 
                cv2.circle(canvas, (x, y), face_radius, (255, 255, 255), thickness=-1)
    
    return canvas


class BodyRatioMapperSDPoseRender:
    """
    SDPose (WholeBody) Render Node. 
    Output is aligned with SDPose-OOD rendering behavior.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT", {"tooltip": "SDPose format POSE_KEYPOINT data"}),
            },
            "optional": {
                "resolution_x": ("INT", {"default": -1, "min": -1, "max": 12800, "tooltip": "Output width (-1 for original)"}),
                "score_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "scale_for_xinsir": ("BOOLEAN", {"default": False, "tooltip": "Apply adaptive thickness scaling used in Xinsir ControlNet"}),
                "stick_width": ("INT", {"default": 4, "min": 1, "max": 32, "tooltip": "Skeleton line width"}),
                "face_point_size": ("INT", {"default": 3, "min": 1, "max": 16, "tooltip": "Face keypoint radius"}),
                "draw_face": ("BOOLEAN", {"default": True, "label_on": "Draw Face", "label_off": "Hide Face"}),
                "draw_mouth": ("BOOLEAN", {"default": True, "label_on": "Draw Mouth", "label_off": "Hide Mouth"}),
                "draw_hands": ("BOOLEAN", {"default": True, "label_on": "Draw Hands", "label_off": "Hide Hands"}),
                "draw_feet": ("BOOLEAN", {"default": True, "label_on": "Draw Feet", "label_off": "Hide Feet"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "render_img"
    CATEGORY = "BodyRatioMapper"

    def render_img(
        self,
        pose_keypoint,
        resolution_x=-1,
        score_threshold=0.3,
        scale_for_xinsir=False,
        stick_width=4,
        face_point_size=3,
        draw_face=True,
        draw_mouth=True,
        draw_hands=True,
        draw_feet=True,
    ):
        if not pose_keypoint:
            return (torch.zeros((1, 512, 512, 3)),)

        batch_frames = []
        
        for frame in pose_keypoint:
            orig_w = frame.get('canvas_width', 512)
            orig_h = frame.get('canvas_height', 768)
            
            # Resolution handling.
            if resolution_x > 0:
                scale = resolution_x / orig_w
                target_w = resolution_x
                target_h = int(orig_h * scale)
            else:
                target_w = orig_w
                target_h = orig_h
                scale = 1.0
                
            # Initialize black canvas.
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            people = frame.get('people', [])
            
            for person in people:
                # Build standard 134-point arrays.
                full_kpts = np.zeros((134, 2))
                full_scores = np.zeros(134)
                
                # Fill arrays in SDPose standard order.
                parts = [
                    ('pose_keypoints_2d', 0, 18),
                    ('foot_keypoints_2d', 18, 6),
                    ('face_keypoints_2d', 24, 68),
                    ('hand_right_keypoints_2d', 92, 21),
                    ('hand_left_keypoints_2d', 113, 21)
                ]
                
                for part_name, start_idx, count in parts:
                    data = person.get(part_name, [])
                    if not data: continue
                    arr = np.array(data).reshape(-1, 3)
                    num = min(len(arr), count)
                    if num > 0:
                        coords = arr[:num, :2].copy()
                        # Automatically handle coordinate scaling.
                        if np.max(coords) <= 1.5: # Normalized coordinates
                            coords[:, 0] *= target_w
                            coords[:, 1] *= target_h
                        else: # Pixel coordinates
                            coords *= scale
                        
                        full_kpts[start_idx:start_idx+num] = coords
                        full_scores[start_idx:start_idx+num] = arr[:num, 2]

                # Render directly on RGB canvas with RGB colors.
                canvas = draw_sdpose_wholebody_standard(
                    canvas, full_kpts, full_scores, 
                    threshold=score_threshold, 
                    scale_for_xinsir=scale_for_xinsir,
                    stick_width=stick_width,
                    face_point_size=face_point_size,
                    draw_face=draw_face,
                    draw_mouth=draw_mouth,
                    draw_hands=draw_hands,
                    draw_feet=draw_feet,
                )

            # Canvas is already RGB and ComfyUI expects RGB, so no BGR->RGB conversion is needed.
            batch_frames.append(torch.from_numpy(canvas.astype(np.float32) / 255.0))
            
        if not batch_frames:
             return (torch.zeros((1, 512, 512, 3)),)
             
        return (torch.stack(batch_frames),)



