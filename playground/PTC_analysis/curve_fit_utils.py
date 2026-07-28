import numpy as np

class CurveFitHelper:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, f"_{k}", v)

    def do_fit(self, xdat: np.ndarray, ydat: np.ndarray):
        pass

    @property
    def errors(self):
        pass

    def calc_residuals(self, xdat):
        pass
