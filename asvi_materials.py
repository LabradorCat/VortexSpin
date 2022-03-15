import numpy as np


class NanoBar:
    def __init__(self, xpos, ypos, zpos, xmag, ymag, zmag,
                 hc_m=0.03, hc_v=0.02, hc_std=0.02, hc_v_std=0.03,
                 bar_l=600e-9, bar_w=150e-9, bar_t=20e-9,
                 magnetisation=800e3, type='macrospin'):
        assert type in ('macrospin', 'vortex')
        self.type = type
        self.pos = np.array([xpos, ypos, zpos], dtype=float)
        self.mag = np.array([xmag, ymag, zmag], dtype=float)
        self.unit_vector = self.mag / np.linalg.norm(self.mag)
        self.magnetisation = magnetisation
        # Coercive field
        self.hc_m = hc_m            # macrospin mean coercive field
        self.hc_v = hc_v            # vortex mean coercive field
        self.hc_std = hc_std        # percentage standard deviation in macrospin coercive field
        self.hc_v_std = hc_v_std    # percentage standard deviation in vortex coercive field
        self.hc = np.random.normal(loc=hc_m, scale=hc_std * hc_m, size=None)
        if self.type == 'vortex':
            self.set_hc(hc_v, hc_v_std)
        self.hc_bias = (self.hc - self.hc_m) / self.hc_m
        # Local field
        self.h_local = None
        # Material Properties
        self.bar_l = bar_l
        self.bar_w = bar_w
        self.bar_t = bar_t

    def __repr__(self):
        return f'{self.type}({self.pos})'

    # CLASS GETTER AND SETTER
    def set_hc(self, hc_u, hc_std):
        hc_new = np.random.normal(loc=hc_u, scale=hc_u*hc_std, size=None)
        # update class properties
        self.hc = hc_new

    # CLASS METHODS
    def flip(self):
        self.mag = np.negative(self.mag)
        self.unit_vector = np.negative(self.unit_vector)

    def set_vortex(self):
        if self.type == 'macrospin':
            self.mag = np.array([0, 0, 0])
            self.type = 'vortex'
            self.set_hc(self.hc_v, self.hc_v_std)
        else:
            print('Nonabar already in vortex state')

    def set_macrospin(self):
        if self.type == 'vortex':
            self.mag = self.unit_vector
            self.type = 'macrospin'
            self.set_hc(self.hc_m, self.hc_std)
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
        return f'Vertex({self.v_c})'

