* ------------------------------------------------------------
* Focused event-based HSPICE testbench for MAC4bit
* Goal:
*   - Measure one "reasonable" propagation delay (50%-50%) for a single, controlled toggle
*   - Compute energy and average power only in a short window around that toggle
*
* Assumptions:
*   - DUT subckt name: MAC4bit
*   - DUT ports are identical to your previous MAC4bit_from_verilog.sp:
*       a_0..a_3, b_0..b_3, c_0..c_7, out_0..out_7
*   - VDD/VSS nets in this testbench are the supply nets used inside the DUT
* ------------------------------------------------------------

.TITLE '* MAC4bit: single-toggle delay + event energy/power'

* ====== User knobs ======
.param VDD   = 0.75
.param TR     = 50p
.param TF     = 50p
.param TTOG   = 1n          * toggle time (a_0 edge starts here)
.param TPRE   = 200p        * window starts TTOG-TPRE
.param TPOST  = 800p        * window ends   TTOG+TPOST
.param VTH    = 'VDD/2'

* analysis window for energy/power
.param TWIN0  = 'TTOG-TPRE'
.param TWIN1  = 'TTOG+TPOST'
.param DTWIN  = 'TWIN1-TWIN0'

* ====== Models / libraries ======
* (keep your own .lib lines here; below are placeholders)
.include 'va_model_wrapper_new.inc'

* ====== Standard-cell CDL (mos14n/mos14p adapted) ======
.include 'and_subckt_nn.inc'
.include 'buf_subckt_nn_cv.inc'
.include 'inv_subckt_nn_cv.inc'
.include 'nand_subckt_nn.inc'
.include 'nor_subckt_nn.inc'
.include 'or_subckt_nn.inc'
.include 'xnor_subckt_nn.inc'
.include 'xor_subckt_nn.inc'

* ====== DUT netlist converted from Verilog ======
* IMPORTANT: keep this filename the SAME as your previous flow so you can reuse include paths.
.include 'MAC4bit_from_verilog.sp'

* ====== Supplies ======
VDD  VDD  0  'VDD'
VSS  VSS  0  0

* ====== Controlled input scenario ======
* Make MAC behave like a simple buffer from a_0 to out_0:
*   set b_0=1, b_1..b_3=0, a_1..a_3=0, c[7:0]=0
* then out_0 should toggle with a_0 (under normal MAC semantics).
*
* Single toggle on a_0: 0 -> 1 at TTOG, then 1 -> 0 at (TTOG + 1ns)
Va0 a_0 0 PULSE(0 'VDD' 'TTOG' 'TR' 'TF' 1n 2n)

Va1 a_1 0 0
Va2 a_2 0 0
Va3 a_3 0 0

Vb0 b_0 0 'VDD'
Vb1 b_1 0 0
Vb2 b_2 0 0
Vb3 b_3 0 0

Vc0 c_0 0 0
Vc1 c_1 0 0
Vc2 c_2 0 0
Vc3 c_3 0 0
Vc4 c_4 0 0
Vc5 c_5 0 0
Vc6 c_6 0 0
Vc7 c_7 0 0

* Optional small output load (helps make waveform realistic & stable)
Cout0 out_0 0 4f

* ====== DUT instantiation ======
XU_MAC4 a_0 a_1 a_2 a_3  b_0 b_1 b_2 b_3  c_0 c_1 c_2 c_3 c_4 c_5 c_6 c_7  out_0 out_1 out_2 out_3 out_4 out_5 out_6 out_7  MAC4bit

* ====== Transient ======
.option post=2 psf=2 probe measout=1 measdgt=8 measform=1 redefsub=2
.option nomod
*.option ingold=2
.tran 1p 'TTOG+TPOST+1n'   * simulate a little beyond the window (includes both edges)

* ====== Delay measurement (pick ONE reasonable path) ======
* Measure delay from a_0 rising -> out_0 rising (50%-50%)
.measure tran tpd_a0_out0_r  TRIG v(a_0)  VAL='VTH' RISE=1
+                            TARG v(out_0) VAL='VTH' RISE=1

* Measure delay from a_0 falling -> out_0 falling (50%-50%)
.measure tran tpd_a0_out0_f  TRIG v(a_0)  VAL='VTH' FALL=1
+                            TARG v(out_0) VAL='VTH' FALL=1

.measure tran tpd_avg param='(tpd_a0_out0_r + tpd_a0_out0_f)/2'

* ====== Energy / power only in a short window around the first edge ======
* By HSPICE sign convention, I(VDD) is often negative when supplying power.
* Use -V*I so energy/power are positive.
.measure tran e_event  INTEG par('-V(VDD)*I(VDD)') FROM='TWIN0' TO='TWIN1'
.measure tran p_event_avg  PARAM='e_event/DTWIN'

* (Optional) estimate leakage power in a quiet window BEFORE the toggle
.param TLEAK0 = 'TTOG-800p'
.param TLEAK1 = 'TTOG-300p'
.param DTLEAK = 'TLEAK1-TLEAK0'
.measure tran e_leak  INTEG par('-V(VDD)*I(VDD)') FROM='TLEAK0' TO='TLEAK1'
.measure tran p_leak_avg  PARAM='e_leak/DTLEAK'

* Dynamic energy in the event window (subtract leakage baseline)
.measure tran e_dyn  PARAM='e_event - p_leak_avg*DTWIN'
.measure tran p_dyn_avg  PARAM='e_dyn/DTWIN'

.probe V(a_0) V(out_0)

.end
