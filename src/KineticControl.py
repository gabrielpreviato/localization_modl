import math

import numpy as np
import pathlib

from src.ObjectClasses import ObjectClasses


class KineticControl(object):
    def __init__(self, file="/tmp/walk"):
        self.file = file

        self.lin_move = math.sqrt(1.3294912578616352 ** 2 + 0.08478180251572327 ** 2)
        self.rot = -0.00010062893081760997

    def do_movement(self):
        open(self.file, 'a').close()
        return self.lin_move, self.rot
