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

__all__ = [
    # Proportion transfer node
    "BodyRatioMapperProportionTransfer",

    # Render node
    "BodyRatioMapperSDPoseRender",

    # Bone scale node
    "BodyRatioMapperSDPoseBoneScale",

    # Translate node
    "BodyRatioMapperSDPoseTranslate",
]

