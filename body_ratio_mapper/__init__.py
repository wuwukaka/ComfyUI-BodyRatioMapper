# Copyright (C) 2026 wuwukasi (wuwukaka)
# SPDX-License-Identifier: GPL-3.0-only

"""
BodyRatioMapper nodes package
Provides organized node classes for pose rendering and transformation
"""

# Import proportion transfer node class
from .proportion_transfer_node import (
    BodyRatioMapperProportionTransfer,
)

# Import render node class
from .render_nodes import (
    BodyRatioMapperSDPoseRender,
)

# Import bone scale node class
from .bone_scale_node import (
    BodyRatioMapperSDPoseBoneScale,
    BodyRatioMapperSDPoseTranslate,
)

# Import interpolation node class
from .interpolation_node import (
    BodyRatioMapperSDPoseInterpolate,
)

__all__ = [
    # Proportion transfer node
    "BodyRatioMapperProportionTransfer",

    # Render node
    "BodyRatioMapperSDPoseRender",

    # Bone scale node
    "BodyRatioMapperSDPoseBoneScale",

    # Translate node
    "BodyRatioMapperSDPoseTranslate",

    # Interpolation node
    "BodyRatioMapperSDPoseInterpolate",
]

