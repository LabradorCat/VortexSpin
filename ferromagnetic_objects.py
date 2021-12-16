import time

import numpy as np
import time as t


class MacroSpin():
    """
    Object class for macrospin, a nano-scale ferromagnetic material that has a single-domain magnetic dipole
    """
    def __init__(self, pos, mag, Hc_mean = 0.025, Hc_std = 0.05,
                 bar_length = 220e-9, bar_thickness = 25e-9, bar_width = 80e-9):
        assert hasattr(pos, '__len__')
        assert hasattr(pos, '__len__')
        assert len(pos) == len(mag)
        # Positional properties
        self.pos = np.array(pos)
        # Dimensional properties
        self.bar_l = bar_length
        self.bar_w = bar_width
        self.bar_t = bar_thickness
        # Magnetic properties
        self.mag = np.array(mag)
        self.Hc_mean = Hc_mean
        self.Hc_std = Hc_std
        self.Hc = np.random.normal(loc=Hc_mean, scale=Hc_std * Hc_mean, size=None)

    # CLASS GETTER & SETTER
    def pos(self):
        return self.pos

    def mag(self):
        return self.mag

    def bar_length(self):
        return self.bar_l

    def bar_width(self):
        return self.bar_w

    def bar_thickness(self):
        return self.bar_t
