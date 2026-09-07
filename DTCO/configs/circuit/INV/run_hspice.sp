* HSPICE deck for INV - 1-cycle (slew / 8*slew / slew / 8*slew)
* Subckt: INV1_90S9T20R

.include 'va_model_wrapper_new.inc'
.include 'inv_subckt_nn_cv.inc'

* =======================
* Global parameters
* =======================
.param VDD   = 0.7
.param CL    = 5.70635e-16

* === input slew definition ===
.param slew  = 1e-10
.param TR    = 'slew'
.param TF    = 'slew'
.param PW    = '8*slew'
.param PERIOD= '18*slew'

XU1 I VDD VNW VPW VSS ZN INV1_90S9T20R

VVDD VDD 0 'VDD'
VVSS VSS 0 0
VVNW VNW 0 'VDD'
VVPW VPW 0 0

CLOAD ZN 0 CL

* input waveform
VI I 0 PULSE(0 'VDD' 0 'TR' 'TF' 'PW' 'PERIOD')

* simulate exactly 1 cycle (parametric)
.tran 1p 'PERIOD'

.option post=1 psf=2 probe measout=1 measdgt=8 measform=1 redefsub=2
.option dccap=1 nomod method=gear runlvl=6 accurate=1
.option ingold=2
.TEMP 25

.print tran V(I) V(ZN) I(VVDD) I(VVSS)

* =======================
* Delay measurements
* =======================
.measure tran TPLH_rising TRIG V(I)  VAL='0.5*VDD' FALL=1
+                    TARG V(ZN) VAL='0.5*VDD' RISE=1
.measure tran TPHL_rising TRIG V(I)  VAL='0.5*VDD' RISE=1
+                    TARG V(ZN) VAL='0.5*VDD' FALL=1
.measure tran TPD_AVG PARAM='(TPLH_rising + TPHL_rising)/2'

* =======================
* 1-cycle power / energy
* =======================
.measure tran PAVG_1C AVG   '(-V(VDD)*I(VVDD))' FROM=0 TO='PERIOD'
*.measure tran E_1C    INTEG '(-V(VDD)*I(VVDD))' FROM=0 TO='PERIOD'

.end
