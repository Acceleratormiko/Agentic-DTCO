* BUFV1_90S9T20R using 7-pin wrapper (mos14n_cv / mos14p_cv), VA/NN only.
* Polarity fix: first-stage devices (X7,X9) have circuit drain at lower potential than source;
* VA expects V(d,s)>0 for positive Id. So we swap d/s at call so VA sees d=node3, s=rail.
.SUBCKT BUFV1_90S9T20R I VDD VNW VPW VSS Z
* X7: nmos circuit d=VSS g=I s=3 → call d=3 s=VSS so VA Id pulls 3 down
X7 3 I VSS VSS cgso_n7 cgdo_n7 cggo_n7 mos14n_cv
* X8: nmos d=Z g=3 s=VSS b=VSS (Vds=Z≥0, no swap)
X8 Z 3 VSS VSS cgso_n8 cgdo_n8 cggo_n8 mos14n_cv
* X9: pmos circuit d=VDD g=I s=3 → call d=3 s=VDD so VA Id (negative) = current VDD→3, pulls 3 up
X9 3 I VDD VDD cgso_p9 cgdo_p9 cggo_p9 mos14p_cv
* X10: pmos d=Z g=3 s=VDD (no swap)
X10 Z 3 VDD VDD cgso_p10 cgdo_p10 cggo_p10 mos14p_cv
Rcgso_n7 cgso_n7 0 1e12
Rcgdo_n7 cgdo_n7 0 1e12
Rcggo_n7 cggo_n7 0 1e12
Rcgso_n8 cgso_n8 0 1e12
Rcgdo_n8 cgdo_n8 0 1e12
Rcggo_n8 cggo_n8 0 1e12
Rcgso_p9 cgso_p9 0 1e12
Rcgdo_p9 cgdo_p9 0 1e12
Rcggo_p9 cggo_p9 0 1e12
Rcgso_p10 cgso_p10 0 1e12
Rcgdo_p10 cgdo_p10 0 1e12
Rcggo_p10 cggo_p10 0 1e12
.ENDS BUFV1_90S9T20R
