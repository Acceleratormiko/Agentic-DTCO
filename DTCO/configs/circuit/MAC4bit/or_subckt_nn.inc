* OR2V1_90S9T20R for VA/NN: same topology as CDL, d/s swapped where needed.
* CDL: X5(4 A1 VSS) X6(VSS A2 4) X7(Z 4 VSS) X8(11 A1 4 VDD) X9(VDD A2 11) X10(Z 4 VDD).
* X6: swap to 4 A2 VSS VSS. X9: swap to 11 A2 VDD VDD.
* X8: when A1 low current 11->4; need I(4,11)<0 so d=4 s=11 -> 4 A1 11 VDD.
.SUBCKT OR2V1_90S9T20R A1 A2 VDD VNW VPW VSS Z
X5 4 A1 VSS VSS mos14n
X6 4 A2 VSS VSS mos14n
X7 Z 4 VSS VSS mos14n
X8 4 A1 11 VDD mos14p
X9 11 A2 VDD VDD mos14p
X10 Z 4 VDD VDD mos14p
.ENDS OR2V1_90S9T20R
