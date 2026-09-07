* XOR2V1_90S9T20R for VA/NN: same topology as CDL, d/s swapped where needed.
* CDL: X11(6 A2 VSS VSS)..X22(Z 4 VDD VDD). NMOS/PMOS VA: d,s so I(d,s) polarity = current source->drain.
* X14: VSS 6 20 -> swap to d=20 s=VSS. X20: VDD 6 10 -> swap to d=10 s=VDD.
* X19: CDL 10 7 4 VDD => when 7 low we need 4 charged from 10 (current 10->4), so d=4 s=10 => 4 7 10 VDD.
.SUBCKT XOR2V1_90S9T20R A1 A2 VDD VNW VPW VSS Z
X11 6 A2 VSS VSS mos14n
X12 4 7 6 VSS mos14n
X13 20 A1 4 VSS mos14n
X14 20 6 VSS VSS mos14n
X15 7 A1 VSS VSS mos14n
X16 Z 4 VSS VSS mos14n
X17 6 A2 VDD VDD mos14p
X18 4 A1 6 VDD mos14p
X19 4 7 10 VDD mos14p
X20 10 6 VDD VDD mos14p
X21 7 A1 VDD VDD mos14p
X22 Z 4 VDD VDD mos14p
.ENDS XOR2V1_90S9T20R
