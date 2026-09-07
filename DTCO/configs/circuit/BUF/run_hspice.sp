* HSPICE deck for BUF - 1-cycle (VA/NN model)
* Subckt: BUFV1_90S9T20R

.include 'va_model_wrapper_new.inc'
* BUF with 7-pin wrapper (mos14n_cv/mos14p_cv), no CDL to avoid duplicate subckt
.include 'buf_subckt_nn_cv.inc'

* =======================
* Global parameters (edit here only)
* =======================
.param VDD   = 0.7
.param CL    = 6.36414e-16

* waveform per your sketch:
* rise = slew, high = 8*slew, fall = slew, low = 8*slew  -> PERIOD = 18*slew
.param slew  = 1e-10
.param TR    = 'slew'
.param TF    = 'slew'
.param PW    = '8*slew'
.param PERIOD= '18*slew'

* =======================
* DUT and supplies
* =======================
XU1 I VDD VNW VPW VSS Z BUFV1_90S9T20R

VVDD VDD 0 'VDD'
VVSS VSS 0 0
VVNW VNW 0 'VDD'
VVPW VPW 0 0

CLOAD Z 0 CL

* input pulse (non-inverting BUF)
VI I 0 PULSE(0 'VDD' 0 'TR' 'TF' 'PW' 'PERIOD')

* simulate exactly 1 cycle
.tran 1p 'PERIOD'

.option post=1 psf=2 probe measout=1 measdgt=8 measform=1 redefsub=2
.option dccap=1 nomod method=gear runlvl=6 accurate=1
.option ingold=2
.TEMP 25

.print tran V(I) V(Z) I(VVDD)

* =======================
* Delay measurements (BUF: input and output same polarity)
* =======================
.measure tran TPLH_rising TRIG V(I) VAL='0.5*VDD' RISE=1
+                    TARG V(Z) VAL='0.5*VDD' RISE=1
.measure tran TPHL_rising TRIG V(I) VAL='0.5*VDD' FALL=1
+                    TARG V(Z) VAL='0.5*VDD' FALL=1
.measure tran TPD_AVG PARAM='(TPLH_rising + TPHL_rising)/2'

* =======================
* 1-cycle power / energy (from VDD source)
* =======================
.measure tran PAVG_1C AVG   '(-V(VDD)*I(VVDD))' FROM=0 TO='PERIOD'
*.measure tran E_1C    INTEG '(-V(VDD)*I(VVDD))' FROM=0 TO='PERIOD'

.end
