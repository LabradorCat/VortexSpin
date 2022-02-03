import numpy as np




class NanoBar:
    """
    TODO: class docstring
    """
    abc = 1
    def __init__(self, xpos, ypos, zpos, xmag, ymag, zmag,
                hc_m=0.03, hc_v=0.02, hc_s=0.05, bar_l=220e-9, bar_w=25e-9, bar_t=80e-9,
                type='macrospin'):
        """
        """
        if type not in ['macrospin', 'vortex']:
            raise TypeError('NanoBar is only allowed to be initiated as macrospin or vortex')
        # Vector properties
        print(abc)
        self._pos = np.array([xpos, ypos, zpos], dtype=float)
        self._mag = np.array([xmag, ymag, zmag], dtype=float)
        self._uni = np.array([xmag, ymag, zmag], dtype=float)
        # Dimensional Properties
        self._bar_l = bar_l
        self._bar_w = bar_w
        self._bar_t = bar_t
        self._type = type
        # Magnetic properties
        self._hc_m = hc_m
        self._hc_v = hc_v
        self._hc_s = hc_s
        self._hc_macrospin = np.random.normal(loc=hc_m, scale=hc_s * hc_m, size=None)
        self._hc_vortex = np.random.normal(loc=hc_v, scale=hc_s * hc_m, size=None)
        if type == 'macrospin':
            self._hc = self._hc_macrospin
        elif type == 'vortex':
            self._hc = self._hc_vortex
        # Nanobar parameters represented in np array
        self._parameters = np.array([xpos, ypos, zpos, xmag, ymag, zmag, self._hc, 0], dtype=float)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._type}, pos {self._pos}, mag {self._mag})"

    def __array__(self, dtype=None):
        return self._parameters

    # CLASS GETTER & SETTER
    def pos(self):
        return self._pos

    def mag(self):
        return self._mag

    def set_mag(self, mag=None, xmag=None, ymag=None, zmag=None):
        """
        TODO: docstrings for set_mag method
        """
        if len(mag) == 3:
            self._mag = np.array(mag, dtype=float)
        elif mag is None: # change magnetisation vector elements individually
            if xmag is not None:
                self._mag[0] = xmag
            if ymag is not None:
                self._mag[1] = ymag
            if zmag is not None:
                self._mag[2] = zmag
        else:
            raise TypeError('Input magnetisation not supported! Requires array-like object with length 3')

    def unit_vector(self):
        return self._uni

    def hc(self):
        return self._hc

    def set_hc(self, hc):
        self._hc = hc

    def get_type(self):
        return self._type

    # MACROSPIN-VORTEX CONVERTER
    def flip(self):
        mag = np.negative(self.mag())
        self.set_mag(mag)

    def set_vortex(self, random_hc = False):
        if self._type == 'macrospin':
            self.set_mag([0, 0, 0])
            if random_hc:
                hc = np.random.normal(loc=self._hc_v, scale=self._hc_s * self._hc_v, size=None)
            else:
                hc = self._hc_vortex
            self.set_hc(hc)
            self._type = 'vortex'

    def set_macrospin(self, random_hc = False, reverse_direction = False):
        if self._type == 'vortex':
            if random_hc:
                hc = np.random.normal(loc=hc_m, scale=hc_s * hc_m, size=None)
            else:
                hc = self._hc_macrospin
            self.set_hc(hc)
            if reverse_direction:
                mag = np.negative(self.unit_vector())
            else:
                mag = self.unit_vector()
            self.set_mag(mag)
            self._type = 'macrospin'


class Vertex:
    """
    Class of a virtual vertex object in ASVI
    """
    def __init__(self, xpos, ypos, zpos, vc):
        self._pos = np.array([xpos, ypos, zpos], dtype=float)
        self._mag = np.array([0, 0, 0], dtype=float)
        self._vc = vc
        self._type = 'vertex'
        self._parameters = np.array([xpos, ypos, zpos, 0, 0, 0, vc, 1], dtype=float)

    def __repr__(self):
        return f"{self.__class__.__name__}(pos {self._pos}, vertex charge {self._vc})"

    def __array__(self, dtype=None):
        return self._parameters


