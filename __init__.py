# Copyright (C) 2026 wuwukasi (wuwukaka)
# SPDX-License-Identifier: GPL-3.0-only

# Import nodes from nodes.py
from .nodes import NODE_CLASS_MAPPINGS
from .nodes import NODE_DISPLAY_NAME_MAPPINGS

# Web directory for JavaScript files
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
