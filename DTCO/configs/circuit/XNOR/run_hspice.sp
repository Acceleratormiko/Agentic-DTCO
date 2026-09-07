* =========================================
* HSPICE deck for XNOR2 - merged two arcs (VA/NN model)
* Arc#1: A1 toggles, A2=1  -> "A_rising_B1"
* Arc#2: A2 toggles, A1=1  -> "B_rising_A1"
* Keep model/include/cell names unchanged
* =========================================

.include 'va_model_wrapper_new.inc'
.include 'xnor_subckt_nn.inc'

* =======================
* Global parameters (edit here only)
* =======================
.param VDD   = 0.7
.param CL    = 4.18379e-15

* Use slew as pass-in parameter (default uses your original TR/TF value)
.param slew  = 7e-11
.param TR    = 'slew'
.param TF    = 'slew'

* waveform rule (same as your previous merge rule):
* rise = slew, high = 8*slew, fall = slew, low = 8*slew  => PERIOD = 18*slew
.param PW     = '8*slew'
.param PERIOD = '18*slew'

* tiny epsilon to avoid ambiguity at the boundary
.param teps = 1p

* =======================
* DUT (do NOT change model/cell naming)
* =======================
XU1 A1 A2 VDD VNW VPW VSS ZN XNOR2V1_90S9T20R

VVDD VDD 0 0.7
VVSS VSS 0 0
VVNW VNW 0 0.7
VVPW VPW 0 0

CLOAD ZN 0 4.18379e-15

* =======================
* Input stimulus (PWL, one source per node; two windows)
* Window#1: [0, PERIOD]        A1 toggles, A2 held at 1
* Window#2: [PERIOD, 2PERIOD]  A2 toggles, A1 held at 1
* =======================

* A1: toggle in 1st period, then held at 1 in 2nd period
VA1 A1 0 PWL(
+ 0                    0
+ 'TR'                 'VDD'
+ 'TR+PW'              'VDD'
+ 'TR+PW+TF'           0
+ 'PERIOD'             0
+ 'PERIOD+teps'        'VDD'
+ '2*PERIOD'           'VDD'
+ )

* A2: held at 1 in 1st period, then toggle in 2nd period
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
* Delay measurements (keep your original measure names)
* NAND (other input = 1):
*   input RISE -> output FALL  (TPHL)
*   input FALL -> output RISE  (TPLH)
* Window-limited to avoid arc mixing
* =================================================

* --- Arc#1: A_rising_B1 (A1 toggles, A2=1) in [0, PERIOD] ---
.measure tran TPLH_A_rising_B1 TRIG V(A1) VAL='0.5*VDD' RISE=1
+ TARG V(ZN) VAL='0.5*VDD' RISE=1
+ FROM=0 TO='PERIOD'

.measure tran TPHL_A_rising_B1 TRIG V(A1) VAL='0.5*VDD' FALL=1
+ TARG V(ZN) VAL='0.5*VDD' FALL=1
+ FROM=0 TO='PERIOD'

* --- Arc#2: B_rising_A1 (A2 toggles, A1=1) in [PERIOD, 2*PERIOD] ---
.measure tran TPLH_B_rising_A1 TRIG V(A2) VAL='0.5*VDD' RISE=1
+ TARG V(ZN) VAL='0.5*VDD' RISE=1
+ FROM='PERIOD+teps' TO='2*PERIOD'

.measure tran TPHL_B_rising_A1 TRIG V(A2) VAL='0.5*VDD' FALL=1
+ TARG V(ZN) VAL='0.5*VDD' FALL=1
+ FROM='PERIOD+teps' TO='2*PERIOD'

* =================================================
* Per-arc average delay (added)
* =================================================
.measure tran TPD_A_rising_B1 PARAM='(TPLH_A_rising_B1 + TPHL_A_rising_B1)/2'
.measure tran TPD_B_rising_A1 PARAM='(TPLH_B_rising_A1 + TPHL_B_rising_A1)/2'

* =================================================
* Energy / average power per arc (one PERIOD window)
* P(t) = -V(VDD)*I(VVDD) -> positive consumed power
* =================================================
.measure tran E_A_rising_B1 INTEG '(-V(VDD)*I(VVDD))' FROM=0 TO='PERIOD'
.measure tran E_B_rising_A1 INTEG '(-V(VDD)*I(VVDD))' FROM='PERIOD+teps' TO='2*PERIOD'

.measure tran PAVG_A_rising_B1 PARAM='E_A_rising_B1 / PERIOD'
.measure tran PAVG_B_rising_A1 PARAM='E_B_rising_A1 / PERIOD'

* =================================================
* Averages across the two arcs (added)
* =================================================
.measure tran TPLH_AVG  PARAM='(TPLH_A_rising_B1 + TPLH_B_rising_A1)/2'
.measure tran TPHL_AVG  PARAM='(TPHL_A_rising_B1 + TPHL_B_rising_A1)/2'
.measure tran TPD_AVG   PARAM='(TPLH_AVG + TPHL_AVG)/2'
*.measure tran E_AVG     PARAM='(E_A_rising_B1 + E_B_rising_A1)/2'
.measure tran PAVG_1C  PARAM='(PAVG_A_rising_B1 + PAVG_B_rising_A1)/2'

.end
