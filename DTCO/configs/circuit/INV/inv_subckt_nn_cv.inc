* INV1_90S9T20R using 7-pin wrapper (mos14n_cv / mos14p_cv), VA/NN only.
.SUBCKT INV1_90S9T20R I VDD VNW VPW VSS ZN
* NMOS: d=ZN g=I s=VSS b=VSS; cap outputs to internal nodes
X5 ZN I VSS VSS cgso_n1 cgdo_n1 cggo_n1 mos14n_cv
* PMOS: d=ZN g=I s=VDD b=VDD
X6 ZN I VDD VDD cgso_p1 cgdo_p1 cggo_p1 mos14p_cv
Rcgso_n1 cgso_n1 0 1e12
Rcgdo_n1 cgdo_n1 0 1e12
Rcggo_n1 cggo_n1 0 1e12
Rcgso_p1 cgso_p1 0 1e12
Rcgdo_p1 cgdo_p1 0 1e12
Rcggo_p1 cggo_p1 0 1e12
.ENDS INV1_90S9T20R
