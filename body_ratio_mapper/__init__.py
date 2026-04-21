"""
BodyRatioMapper nodes package
Provides organized node classes for pose rendering and transformation
"""

# Import ultimate detector node class
from .proportion_transfer_node import (
    BodyRatioMapperProportionTransfer,
)

# Import render node classes
from .render_nodes import (
    BodyRatioMapperSDPoseRender,
)

__all__ = [
    # Ultimate detector node
    "BodyRatioMapperProportionTransfer",

    # Render nodes
    "BodyRatioMapperSDPoseRender",

]

