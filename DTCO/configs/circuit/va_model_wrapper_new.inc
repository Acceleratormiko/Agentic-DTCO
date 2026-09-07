* VA Model Wrapper for BSIM-compatible interface (using nnmodel_n_type / nnmodel_p_type)
* Maps mos14n/mos14p subcircuit calls to Verilog-A models.
*
* Geometry defaults now live in the generated Verilog-A model itself.
* Do not pass l/nfin/eot from the deck or subckt layer.
*
* NOTE: HspiceTool copies nnmodel_*.va into each circuit work directory,
* so keep local .hdl paths here.

.hdl 'nnmodel_n_type.va'
.hdl 'nnmodel_p_type.va'

.OPTION GMIN=1e-12 ABSTOL=1e-12 RELTOL=1e-3 VNTOL=1e-6
.OPTION CHGTOL=1e-14 MAXWARNS=1000
.OPTION ITL1=1000 ITL2=1000 ITL4=100
.OPTION METHOD=TRAP

* mos14n (NMOS) - same interface as 7nm_ntype_card.mod
* VA module nnmodel_n_type(g, d, s, b, cgs_out, cgd_out, cgg_out, gnd); vs/vb from node voltages V(s,gnd), V(b,gnd)
.subckt mos14n d g s b
X_va_nmos g d s b cgso_n cgdo_n cggo_n 0 nnmodel_n_type
Rcgso_n cgso_n 0 1e12
Rcgdo_n cgdo_n 0 1e12
Rcggo_n cggo_n 0 1e12
.ends mos14n

* mos14n_cv (NMOS for CV deck) - 7 pins: d g s b cgs_out cgd_out cgg_out; VA outputs Cgs,Cgd,Cgg in F for .print ac
.subckt mos14n_cv d g s b cgs_out cgd_out cgg_out enable_junction=0
X_va_nmos g d s b cgs_out cgd_out cgg_out 0 nnmodel_n_type enable_junction=enable_junction
.ends mos14n_cv

* mos14p (PMOS) - same interface as 7nm_ptype_card.mod
.subckt mos14p d g s b
X_va_pmos g d s b cgso_p cgdo_p cggo_p 0 nnmodel_p_type
Rcgso_p cgso_p 0 1e12
Rcgdo_p cgdo_p 0 1e12
Rcggo_p cggo_p 0 1e12
.ends mos14p

* mos14p_cv (PMOS for CV deck) - 7 pins
.subckt mos14p_cv d g s b cgs_out cgd_out cgg_out enable_junction=0
X_va_pmos g d s b cgs_out cgd_out cgg_out 0 nnmodel_p_type enable_junction=enable_junction
.ends mos14p_cv
