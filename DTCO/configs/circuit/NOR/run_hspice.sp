* =========================================
* HSPICE deck for NOR2 - merged two arcs (VA/NN model)
* Arc#1: A toggles, B=0  -> A -> ZN
* Arc#2: B toggles, A=0  -> B -> ZN
* Output: per-arc TPLH/TPHL, per-arc energy/power, and averages
* =========================================

.include 'va_model_wrapper_new.inc'
.include 'nor_subckt_nn.inc'

* =======================
* Global parameters (edit here only)
* =======================
.param VDD   = 0.7
.param CL    = 5.33041e-15

* waveform per your sketch:
* rise = slew, high = 8*slew, fall = slew, low = 8*slew  => PERIOD = 18*slew
.param slew   = 7e-11
.param TR     = 'slew'
.param TF     = 'slew'
.param PW     = '8*slew'
.param PERIOD = '18*slew'

* small time epsilon to avoid ambiguity at window boundary
.param teps = 1p

* =======================
* DUT and supplies
* =======================
XU1 A B VDD VNW VPW VSS ZN NOR2V1_90S9T20R

VVDD VDD 0 'VDD'
VVSS VSS 0 0
VVNW VNW 0 'VDD'
VVPW VPW 0 0

CLOAD ZN 0 'CL'

* =======================
* Input stimulus (PWL, one source per node; two windows)
* Window#1: [0, PERIOD]       A toggles, B held at 0
* Window#2: [PERIOD, 2PERIOD] B toggles, A held at 0
* =======================

* A: pulse in 1st period, then held at 0 in 2nd period
VA A 0 PWL(
+ 0                    0
+ 'TR'                 'VDD'
+ 'TR+PW'              'VDD'
+ 'TR+PW+TF'           0
+ 'PERIOD'             0
+ '2*PERIOD'           0
+ )

* B: held at 0 in 1st period, then pulse in 2nd period
VB B 0 PWL(
+ 0                    0
+ 'PERIOD'             0
+ 'PERIOD+teps'        0
+ 'PERIOD+TR'          'VDD'
+ 'PERIOD+TR+PW'       'VDD'
+ 'PERIOD+TR+PW+TF'    0
+ '2*PERIOD'           0
+ )

* =======================
* Transient: cover two windows
* =======================
.tran 1p '2*PERIOD'

.option post=1 psf=2 probe measout=1 measdgt=8 measform=1 redefsub=2
.option dccap=1 nomod method=gear runlvl=6 accurate=1
.option ingold=2
.TEMP 25

.print tran V(A) V(B) V(ZN) I(VVDD)

* =================================================
* Delay per arc (50%-50%), window-limited
* NOR: input rise -> output fall (TPHL)  when other input = 0
*      input fall -> output rise (TPLH)  when other input = 0
* =================================================

* --- Arc#1: A -> ZN (within [0, PERIOD]) ---
.measure tran TPHL_AZN
+ TRIG V(A)  VAL='0.5*VDD' RISE=1
+ TARG V(ZN) VAL='0.5*VDD' FALL=1
+ FROM=0 TO='PERIOD'

.measure tran TPLH_AZN
+ TRIG V(A)  VAL='0.5*VDD' FALL=1
+ TARG V(ZN) VAL='0.5*VDD' RISE=1
+ FROM=0 TO='PERIOD'

* --- Arc#2: B -> ZN (within [PERIOD, 2*PERIOD]) ---
.measure tran TPHL_BZN
+ TRIG V(B)  VAL='0.5*VDD' RISE=1
+ TARG V(ZN) VAL='0.5*VDD' FALL=1
+ FROM='PERIOD+teps' TO='2*PERIOD'

.measure tran TPLH_BZN
+ TRIG V(B)  VAL='0.5*VDD' FALL=1
+ TARG V(ZN) VAL='0.5*VDD' RISE=1
+ FROM='PERIOD+teps' TO='2*PERIOD'

* =================================================
* Energy / average power per arc (one PERIOD window)
* P(t) = -V(VDD)*I(VVDD) -> positive consumed power
* =================================================

.measure tran E_AZN INTEG '(-V(VDD)*I(VVDD))' FROM=0 TO='PERIOD'
.measure tran E_BZN INTEG '(-V(VDD)*I(VVDD))' FROM='PERIOD+teps' TO='2*PERIOD'

.measure tran PAVG_AZN PARAM='E_AZN / PERIOD'
.measure tran PAVG_BZN PARAM='E_BZN / PERIOD'

* =================================================
* Averages across the two arcs
* =================================================
.measure tran TPHL_AVG  PARAM='(TPHL_AZN + TPHL_BZN)/2'
.measure tran TPLH_AVG  PARAM='(TPLH_AZN + TPLH_BZN)/2'
.measure tran TPD_AVG   PARAM='(TPHL_AVG + TPLH_AVG)/2'
.measure tran PAVG_1C  PARAM='(PAVG_AZN + PAVG_BZN)/2'
*.measure tran E_AVG     PARAM='(E_AZN + E_BZN)/2'

.end
