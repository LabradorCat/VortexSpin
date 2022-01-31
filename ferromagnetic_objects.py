import numpy as np


class NanoBar(np.ndarray):
    """
     Object class for a nano-scale ferromagnetic material in a bar shape
    """
    def __new__(cls, xpos, ypos, zpos, xmag, ymag, zmag,
                hc_m=0.025, hc_s=0.05, bar_l=220e-9, bar_w=25e-9, bar_t=80e-9,
                type='macrospin'):
        hc = np.random.normal(loc=hc_m, scale=hc_s * hc_m, size=None)
        obj = np.asarray([xpos, ypos, zpos, xmag, ymag, zmag, hc, bar_l, bar_w, bar_t]).view(cls)
        obj.type = type
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        # Positional properties
        self.pos = np.array([obj[0], obj[1], obj[2]])
        # Magnetic properties
        self.mag = np.array([obj[3], obj[4], obj[5]])
        self.hc = obj[6]
        self.type = getattr(obj, 'type', 'macrospin')
        # Dimensional properties
        self.bar_l = obj[7]
        self.bar_w = obj[8]
        self.bar_t = obj[9]

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):  # this method is called whenever you use a ufunc
        f = {
            "reduce": ufunc.reduce,
            "accumulate": ufunc.accumulate,
            "reduceat": ufunc.reduceat,
            "outer": ufunc.outer,
            "at": ufunc.at,
            "__call__": ufunc,
        }
        output = NanoBar(f[method](*(i.view(np.ndarray) for i in inputs),**kwargs))
        # convert the inputs to np.ndarray to prevent recursion, call the function, then cast it back as ExampleTensor
        output.__dict__ = self.__dict__  # carry forward attributes
        return output
    # CLASS GETTER & SETTER

