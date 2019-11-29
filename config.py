import os

from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.IMAGE_PATH = "images"
_C.SAVE_PATH = "results"
_C.HEIGHT = 100
_C.ANGLE = 10
_C.OVERLAP = 0.66
