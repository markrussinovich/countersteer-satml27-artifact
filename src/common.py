"""Constants shared by every module: repo root, role list, probe hyperparameters, the sentinel used for span location, and the stub tool schema."""
from __future__ import annotations

import argparse
import asyncio
import glob
import gzip
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch



# REPO ROOT, not the package directory. This file used to sit at the repo root, where
# dirname(__file__) was correct; after the split it is src/, so every path derived
# from ROOT (runs/, logs/, probe pickles) would have silently moved down a level.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLES = ["system", "user", "cot", "assistant", "tool"]
PROBE_C = 5.0e-3          # paper spec
MN_FIT_ROWS = 200_000     # cap rows per probe fit. Fitting is on GPU (fit_logreg_gpu),
                          # so this is bounded by VRAM alongside the model, not by time.
SENTINEL = "ZQXSENTINELXQZ"
TOOL_NAME = "read_record"
_WARNED = {"tools": False}
TOOLS = [{"type": "function", "function": {
    "name": TOOL_NAME, "description": "Read the current record.",
    "parameters": {"type": "object", "properties": {}}}}]
