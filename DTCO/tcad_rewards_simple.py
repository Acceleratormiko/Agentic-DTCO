import math
from typing import Optional

class TCADScoreCalculator:
    """
    Simplified TCADScoreCalculator for DTCO reward.
    """
    def __init__(
        self,
        ion_target: float,
        ioff_target: float,
        ss_target: float,
        ion_tolerate: float,
        ioff_tolerate: float,
        ss_tolerate: float,
        **kwargs
    ):
        self.ion_target = ion_target
        self.ioff_target = ioff_target
        self.ss_target = ss_target
        self.ion_tolerate = ion_tolerate
        self.ioff_tolerate = ioff_tolerate
        self.ss_tolerate = ss_tolerate

    def compute_score(self, ion: float, ioff: float, ss: float) -> float:
        # Ion score: higher is better
        if ion >= self.ion_target:
            s_ion = 1.0
        elif ion <= self.ion_tolerate:
            s_ion = 0.0
        else:
            s_ion = (ion - self.ion_tolerate) / (self.ion_target - self.ion_tolerate)

        # Ioff score: lower is better
        if ioff <= self.ioff_target:
            s_ioff = 1.0
        elif ioff >= self.ioff_tolerate:
            s_ioff = 0.0
        else:
            s_ioff = (self.ioff_tolerate - ioff) / (self.ioff_tolerate - self.ioff_target)

        # SS score: lower is better
        if ss <= self.ss_target:
            s_ss = 1.0
        elif ss >= self.ss_tolerate:
            s_ss = 0.0
        else:
            s_ss = (self.ss_tolerate - ss) / (self.ss_tolerate - self.ss_target)

        # Geometric mean
        return (s_ion * s_ioff * s_ss) ** (1/3)
