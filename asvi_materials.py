import numpy as np


class NanoBar:
    def __init__(self, xpos, ypos, zpos, xmag, ymag, zmag,
                 hc_m=0.03, hc_v=0.02, hc_std=0.02,
                 bar_l=600e-9, bar_w=150e-9, bar_t=20e-9,
                 magnetisation=800e3, type='macrospin'):
        assert type in ('macrospin', 'vortex')
        # Position
        self.pos = np.array([xpos, ypos, zpos], dtype=float)
        # Magnetic Properties
        self.type = type
        self.mag = np.array([xmag, ymag, zmag], dtype=float)
        self.unit_vector = self.mag / np.linalg.norm(self.mag)
        self.magnetisation = magnetisation
        self.hc_m = hc_m        # macrospin mean coersive field
        self.hc_v = hc_v        # vortex mean coersive field
        self.hc_std = hc_std    # percentage standard deviation
        self.hc = np.random.normal(loc=hc_m, scale=hc_std * hc_m, size=None)
        self.hc_bias = (self.hc - self.hc_m) / self.hc_m
        if self.type == 'vortex':
            self.set_hc(hc_v, hc_std)
        # Material Properties
        self.bar_l = bar_l
        self.bar_w = bar_w
        self.bar_t = bar_t

    def __repr__(self):
        return f'NanoBar({self.pos})'

    def __array__(self):
        array = np.append(self.pos, self.mag)
        array = np.append(array, [self.hc, self.bar_l, self.bar_w, self.bar_t])
        return array

    # CLASS GETTER AND SETTER
    def set_mag(self, xmag=None, ymag=None, zmag=None, mag=None):
        """
        Method to set the magnetic dipole of the Nanobar
        Allow user to set mag altogether, or in each direction separately
        """
        if mag is None:
            if xmag is not None: self.mag[0] = xmag
            if ymag is not None: self.mag[1] = ymag
            if ymag is not None: self.mag[2] = zmag
        else:
            assert len(mag) == 3, "Input magnetic array should have 3 indices"
            self.set_mag(xmag=mag[0], ymag=mag[1], zmag=mag[2])

    def set_hc(self, hc_u, hc_std, bias=False):
        if bias:
            hc = hc_u * (1 + self.hc_bias)
        else:
            hc = hc_u
        hc_new = np.random.normal(loc=hc, scale=hc*hc_std, size=None)
        # update class properties
        self.hc = hc_new
        self.hc_m = hc_u
        self.hc_std = hc_std

    # CLASS METHODS
    def flip(self):
        self.set_mag(mag=np.negative(self.mag))
        self.unit_vector = np.negative(self.unit_vector)

    def set_vortex(self):
        if self.type == 'macrospin':
            self.set_mag(0, 0, 0)
            self.set_hc(self.hc_v, self.hc_std)
            self.type = 'vortex'
        else:
            print('Nonabar already in vortex state')

    def set_macrospin(self):
        if self.type == 'vortex':
            self.set_mag(mag = self.unit_vector)
            self.set_hc(self.hc_m, self.hc_std)
            self.type = 'macrospin'
        else:
            print('Nonabar already in macrospin state')


class Vertex:
    def __init__(self, xpos, ypos, zpos, v_c=0, type=1):
        assert type in (1, 2, 3)
        self.pos = np.array([xpos, ypos, zpos], dtype=float)
        self.mag = np.array([0, 0, 0], dtype=float)
        self.hc = None
        self.v_c = v_c
        self.type = type

    def __repr__(self):
        return f'Vertex({self.pos})'

    def __array__(self):
        array = np.append(self.pos, self.mag)
        array = np.append(array, [self.hc, 0, 0, 0])
        return array
    # CLASS GETTER AND SETTER

