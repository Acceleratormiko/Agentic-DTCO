* =========================================
* HSPICE deck for NAND2 - merged two arcs (VA/NN model)
* Arc#1: A1 toggles, A2=1  -> A1 -> ZN
* Arc#2: A2 toggles, A1=1  -> A2 -> ZN
* Output: per-arc TPLH/TPHL, per-arc energy/power, and averages
* =========================================

.include 'va_model_wrapper_new.inc'
.include 'nand_subckt_nn.inc'

* =======================
* Global parameters (edit here only)
* =======================
.param VDD   = 0.7
.param CL    = 4.18379e-15

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
XU1 A1 A2 VDD VNW VPW VSS ZN NAND2V1_90S9T20R

VVDD VDD 0 'VDD'
VVSS VSS 0 0
VVNW VNW 0 'VDD'
VVPW VPW 0 0

CLOAD ZN 0 'CL'

* =======================
* Input stimulus (PWL, one source per node; two windows)
* Window#1: [0, PERIOD]       A1 toggles, A2 held at 1
* Window#2: [PERIOD, 2PERIOD] A2 toggles, A1 held at 1
* =======================

* A1: pulse in 1st period, then held at 1 in 2nd period
VA1 A1 0 PWL(
+ 0                    0
+ 'TR'                 'VDD'
+ 'TR+PW'              'VDD'
+ 'TR+PW+TF'           0
+ 'PERIOD'             0
+ 'PERIOD+teps'        'VDD'
+ '2*PERIOD'           'VDD'
+ )

* A2: held at 1 in 1st period, then pulse in 2nd period
VA2 A2 0 PWL(
+ 0                    'VDD'
+ 'PERIOD'             'VDD'
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

.print tran V(A1) V(A2) V(ZN) I(VVDD)

* =================================================
* Delay per arc (50%-50%), window-limited
* NAND: input rise -> output fall (TPHL)
*       input fall -> output rise (TPLH)
* =================================================

* --- Arc#1: A1 -> ZN (within [0, PERIOD]) ---
.measure tran TPHL_A1ZN
+ TRIG V(A1) VAL='0.5*VDD' RISE=1
+ TARG V(ZN) VAL='0.5*VDD' FALL=1
+ FROM=0 TO='PERIOD'

.measure tran TPLH_A1ZN
+ TRIG V(A1) VAL='0.5*VDD' FALL=1
+ TARG V(ZN) VAL='0.5*VDD' RISE=1
+ FROM=0 TO='PERIOD'

* --- Arc#2: A2 -> ZN (within [PERIOD, 2*PERIOD]) ---
.measure tran TPHL_A2ZN
+ TRIG V(A2) VAL='0.5*VDD' RISE=1
+ TARG V(ZN) VAL='0.5*VDD' FALL=1
+ FROM='PERIOD+teps' TO='2*PERIOD'

.measure tran TPLH_A2ZN
+ TRIG V(A2) VAL='0.5*VDD' FALL=1
+ TARG V(ZN) VAL='0.5*VDD' RISE=1
+ FROM='PERIOD+teps' TO='2*PERIOD'

* =================================================
* Energy / average power per arc (one PERIOD window)
* P(t) = -V(VDD)*I(VVDD) -> positive consumed power
* =================================================

.measure tran E_A1ZN INTEG '(-V(VDD)*I(VVDD))' FROM=0 TO='PERIOD'
.measure tran E_A2ZN INTEG '(-V(VDD)*I(VVDD))' FROM='PERIOD+teps' TO='2*PERIOD'

.measure tran PAVG_A1ZN PARAM='E_A1ZN / PERIOD'
.measure tran PAVG_A2ZN PARAM='E_A2ZN / PERIOD'

* =================================================
* Averages across the two arcs
* =================================================
.measure tran TPHL_AVG  PARAM='(TPHL_A1ZN + TPHL_A2ZN)/2'
.measure tran TPLH_AVG  PARAM='(TPLH_A1ZN + TPLH_A2ZN)/2'
.measure tran TPD_AVG   PARAM='(TPHL_AVG + TPLH_AVG)/2'
.measure tran PAVG_1C  PARAM='(PAVG_A1ZN + PAVG_A2ZN)/2'
*.measure tran E_AVG     PARAM='(E_A1ZN + E_A2ZN)/2'

.end
