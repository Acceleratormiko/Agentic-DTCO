* AND2V1_90S9T20R for VA/NN: same topology as CDL, d/s swapped where needed so VA sees correct polarity.
* VA convention: I(d,s) with Vds=V(d)-V(s); NMOS Id>0 when Vds>0; PMOS use d=circuit_drain, s=circuit_source so I(d,s)<0 = current source->drain.
* l/nfin/eot from VA model defaults.
*
* Pull-down chain: 4 - X5 - 11 - X6 - VSS. When both A1,A2 high, current 4->11->VSS. So X5 drain=4 source=11, X6 drain=11 source=VSS.
* CDL: X5(11 A1 4 VSS) => swap to D=4 S=11. X6(VSS A2 11 VSS) => swap to D=11 S=VSS. X9(VDD A2 4 VDD) => swap to d=4 s=VDD.
.SUBCKT AND2V1_90S9T20R A1 A2 VDD VNW VPW VSS Z
* X5: swap so VA sees D=4 S=11 (current 4->11 when A1 high)
X5 4 A1 11 VSS mos14n
* X6: swap so VA sees D=11 S=VSS (Vds>=0 when on)
X6 11 A2 VSS VSS mos14n
* X7: D=Z S=VSS, OK
X7 Z 4 VSS VSS mos14n
* X8: D=4 S=VDD (I(4,VDD)<0 = VDD->4), OK
X8 4 A1 VDD VDD mos14p
* X9: swap so VA sees d=4 s=VDD (I(4,VDD)<0 = VDD->4)
X9 4 A2 VDD VDD mos14p
* X10: D=Z S=VDD, OK
X10 Z 4 VDD VDD mos14p
.ENDS AND2V1_90S9T20R
