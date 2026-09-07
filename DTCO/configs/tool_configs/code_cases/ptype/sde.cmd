(sde:clear)
(sde:set-process-up-direction "+z")
(sdegeo:set-default-boolean "ABA")

; (1). Parameters
; Design parameter
(define pi (acos -1))  ; Get value of pi
(define Lg 0.018)
(define Lext 0.005)
(define Lsd 0.014)
(define Fh 0.035)
(define Fwt 0.006)
(define Fsp 0.03)
(define theta 0.82)
(define ToxLK 0.5e-3)
(define ToxHK 1.13e-3)
(define num_of_fin 1)
(define Hsti 0.05)
(define Hsub 0.1)
(define Hmetal 3e-3)

; Derived quantities
(define theta_rad (* theta (/ pi 180)))
(define Wsti (- (* 0.5 (- Fsp Fwt)) (* Fh (tan theta_rad))))
(display Wsti)
(define fillet_r (/ Fwt 4))

; Define region string
(define sti_1_name "R.STI_1_")
(define sti_2_name "R.STI_2_")
(define s_name "R.Source_")
(define sext_name "R.SourceExt_")
(define c_name "R.Channel_")
(define d_name "R.Drain_")
(define dext_name "R.DrainExt_")
(define gate_oxide_p1_name "R.GateOxide_LK_1_")
(define gate_oxide_p2_name "R.GateOxide_HK_1_")

; x-Coordinate
(define x1 Lsd)
(define x2 (+ x1 Lext))
(define x3 (+ x2 Lg))
(define x4 (+ x3 Lext))
(define x5 (+ x4 Lsd))

; y-Coordinate for the first fin
(define y1 Fsp)
(define y2 Wsti)
(define y3 (- y1 Wsti))
(define y4 (+ y2 (* Fh (tan theta_rad))))
(define y5 (- y3 (* Fh (tan theta_rad))))
(define y6 (+ y3 (/ ToxLK (cos theta_rad))))
(define y7 (- y2 (/ ToxLK (cos theta_rad))))
(define y8 (+ y5 (* ToxLK (/ (- 1 (sin theta_rad)) (cos theta_rad)))))
(define y9 (- y4 (* ToxLK (/ (- 1 (sin theta_rad)) (cos theta_rad)))))

; A patch for HKMG
(define y10 (+ y6 (/ ToxHK (cos theta_rad))))
(define y11 (- y7 (/ ToxHK (cos theta_rad))))
(define y12 (+ y8 (* ToxHK (/ (- 1 (sin theta_rad)) (cos theta_rad)))))
(define y13 (- y9 (* ToxHK (/ (- 1 (sin theta_rad)) (cos theta_rad)))))

; z-Coordinate
(define z1 Hsub)
(define z2 (+ z1 Hsti))
(define z3 (+ z2 Fh))
(define z4 (+ z3 ToxLK))
(define z5 (+ z2 ToxLK))

; A patch for HKMG
(define z7 (+ z4 ToxHK))
(define z8 (+ z5 ToxHK))
(define z6 (+ z7 Hmetal))

; Doping Information
(define SD_doping 1e20)
(define SD_extension_doping 1e19)
(define C_doping 5e17)
(define Sub_Doping 1e18)
(define SD_dopant "BoronActiveConcentration")
(define substrate_dopant "ArsenicActiveConcentration")

; (2). Structure

(do ((i 0 (+ i 1)))
    ((= i num_of_fin))
    (begin
        (sdegeo:set-default-boolean "ABA")

        ; STI SiO2 layer
        (sdegeo:create-cuboid (position 0 (+ 0 (* i Fsp)) z1) (position x5 (+ y2 (* i Fsp)) z2) "SiO2" (string-append sti_1_name (number->string i)))
        (sdegeo:create-cuboid (position 0 (+ y3 (* i Fsp)) z1) (position x5 (+ y1 (* i Fsp)) z2) "SiO2" (string-append sti_2_name (number->string i)))

        ; Source/Source Extension
        (define S_vertices (list (position 0 (+ y2 (* i Fsp)) z2) 
                                (position 0 (+ y4 (* i Fsp)) z3) 
                                (position 0 (+ y5 (* i Fsp)) z3) 
                                (position 0 (+ y3 (* i Fsp)) z2)))
        (define S_face (sdegeo:create-polygon S_vertices "Silicon" (string-append s_name (number->string i))))
        (sdegeo:sweep S_face (gvector Lsd 0 0))
        (define SExt_vertices (list (position x1 (+ y2 (* i Fsp)) z2) 
                                (position x1 (+ y4 (* i Fsp)) z3) 
                                (position x1 (+ y5 (* i Fsp)) z3) 
                                (position x1 (+ y3 (* i Fsp)) z2)))
        (define SExt_face (sdegeo:create-polygon SExt_vertices "Silicon" (string-append sext_name (number->string i))))
        (sdegeo:sweep SExt_face (gvector Lext 0 0))

        ; Channel
        (define C_vertices (list (position x2 (+ y2 (* i Fsp)) z2) 
                                (position x2 (+ y4 (* i Fsp)) z3) 
                                (position x2 (+ y5 (* i Fsp)) z3) 
                                (position x2 (+ y3 (* i Fsp)) z2)))
        (define C_face (sdegeo:create-polygon C_vertices "Silicon" (string-append c_name (number->string i))))
        (sdegeo:sweep C_face (gvector Lg 0 0))

        ; Drain/Drain Extension
        (define DExt_vertices (list (position x3 (+ y2 (* i Fsp)) z2) 
                                (position x3 (+ y4 (* i Fsp)) z3) 
                                (position x3 (+ y5 (* i Fsp)) z3) 
                                (position x3 (+ y3 (* i Fsp)) z2)))
        (define DExt_face (sdegeo:create-polygon DExt_vertices "Silicon" (string-append dext_name (number->string i))))
        (sdegeo:sweep DExt_face (gvector Lext 0 0))

        (define D_vertices (list (position x4 (+ y2 (* i Fsp)) z2) 
                                (position x4 (+ y4 (* i Fsp)) z3) 
                                (position x4 (+ y5 (* i Fsp)) z3) 
                                (position x4 (+ y3 (* i Fsp)) z2)))
        (define D_face (sdegeo:create-polygon D_vertices "Silicon" (string-append d_name (number->string i))))
        (sdegeo:sweep D_face (gvector Lsd 0 0))

        ; Fillet for the fin
        (define fin_edge_list (list 
            (car (find-edge-id (position (/ x1 2) (+ y4 (* i Fsp)) z3)))
            (car (find-edge-id (position (/ (+ x2 x1) 2) (+ y4 (* i Fsp)) z3)))
            (car (find-edge-id (position (/ (+ x3 x2) 2) (+ y4 (* i Fsp)) z3)))
            (car (find-edge-id (position (/ (+ x4 x3) 2) (+ y4 (* i Fsp)) z3)))
            (car (find-edge-id (position (/ (+ x5 x4) 2) (+ y4 (* i Fsp)) z3)))
            (car (find-edge-id (position (/ x1 2) (+ y5 (* i Fsp)) z3)))
            (car (find-edge-id (position (/ (+ x2 x1) 2) (+ y5 (* i Fsp)) z3)))
            (car (find-edge-id (position (/ (+ x3 x2) 2) (+ y5 (* i Fsp)) z3)))
            (car (find-edge-id (position (/ (+ x4 x3) 2) (+ y5 (* i Fsp)) z3)))
            (car (find-edge-id (position (/ (+ x5 x4) 2) (+ y5 (* i Fsp)) z3)))
        ))
        (sdegeo:fillet fin_edge_list fillet_r)

        ; Gate Oxide part 1
        (sdegeo:set-default-boolean "BAB")
        (define oxide_vertices (list (position x2 (+ y6 (* i Fsp)) z2) 
                                (position x2 (+ y8 (* i Fsp)) z4) 
                                (position x2 (+ y9 (* i Fsp)) z4) 
                                (position x2 (+ y7 (* i Fsp)) z2)))
        (define oxide_face (sdegeo:create-polygon oxide_vertices "InterfacialOxide" (string-append gate_oxide_p1_name (number->string i))))
        (sdegeo:sweep oxide_face (gvector Lg 0 0))

        ; Fillet for the gate oxide
        (define fin_edge_list (list 
            (car (find-edge-id (position (/ (+ x3 x2) 2) (+ y8 (* i Fsp)) z4)))
            (car (find-edge-id (position (/ (+ x3 x2) 2) (+ y9 (* i Fsp)) z4)))
        ))
        (sdegeo:fillet fin_edge_list (+ fillet_r ToxLK))

        ; Gate Oxide part 2
        (sdegeo:set-default-boolean "BAB")
        (define oxide_vertices (list (position x2 (+ y10 (* i Fsp)) z5) 
                                (position x2 (+ y12 (* i Fsp)) z7) 
                                (position x2 (+ y13 (* i Fsp)) z7) 
                                (position x2 (+ y11 (* i Fsp)) z5)))
        (define oxide_face (sdegeo:create-polygon oxide_vertices "HfO2" (string-append gate_oxide_p2_name (number->string i))))
        (sdegeo:sweep oxide_face (gvector Lg 0 0))

        ; Fillet for the gate oxide
        (define fin_edge_list (list 
            (car (find-edge-id (position (/ (+ x3 x2) 2) (+ y12 (* i Fsp)) z7)))
            (car (find-edge-id (position (/ (+ x3 x2) 2) (+ y13 (* i Fsp)) z7)))
        ))
        (sdegeo:fillet fin_edge_list (+ fillet_r ToxHK))

        ; Gate Oxide part 2
        ;(sdegeo:create-cuboid (position x2 (+ 0 (* i Fsp)) z2) (position x3 (+ y1 (* i Fsp)) z5) "InterfacialOxide" (string-append gate_oxide_p2_name (number->string i)))
        ;(display i)
        ;(newline)
    )
)

; Gate Oxide part 1
(sdegeo:set-default-boolean "BAB")
(sdegeo:create-cuboid (position x2 0 z2) (position x3 (* y1 num_of_fin) z5) "InterfacialOxide" "R.GateOxide_LK_2")

; Gate Oxide part 2
(sdegeo:create-cuboid (position x2 0 z5) (position x3 (* y1 num_of_fin) z8) "HfO2" "R.GateOxide_HK_2")

; Substrate
(sdegeo:create-cuboid (position 0 0 0) (position x5 (* y1 num_of_fin) z2) "Silicon" "R.Body")

; Gate contact metal layer
(sdegeo:create-cuboid (position x2 0 z8) (position x3 (* y1 num_of_fin) z6) "Metal" "R.GateContact")

; S/D contact metal layer
(sdegeo:create-cuboid (position 0 0 z2) (position x1 (* y1 num_of_fin) z6) "Metal" "R.SourceContact")
(sdegeo:create-cuboid (position x4 0 z2) (position x5 (* y1 num_of_fin) z6) "Metal" "R.DrainContact")

; S/D spacer layer
(sdegeo:create-cuboid (position x1 0 z2) (position x2 (* y1 num_of_fin) z6) "Si3N4" "R.SourceSpacer")
(sdegeo:create-cuboid (position x3 0 z2) (position x4 (* y1 num_of_fin) z6) "Si3N4" "R.DrainSpacer")

; (3). Contact
; Source
(sdegeo:set-contact (find-face-id (position (/ x1 2) (/ (* y1 num_of_fin) 2) z6)) "source")

; Drain
(sdegeo:set-contact (find-face-id (position (/ (+ x4 x5) 2) (/ (* y1 num_of_fin) 2) z6)) "drain")

; Gate
(sdegeo:set-contact (find-face-id (position (/ (+ x2 x3) 2) (/ (* y1 num_of_fin) 2) z6)) "gate")

; Substrate
(sdegeo:set-contact (find-face-id (position (/ x5 2) (/ (* y1 num_of_fin) 2) 0)) "substrate")

; (4). Doping
; Source Doping
(sdedr:define-constant-profile "DP.source" SD_dopant SD_doping)
(do ((i 0 (+ i 1)))
    ((= i num_of_fin))
    (begin
        (sdedr:define-constant-profile-region (string-append "DDP_S_" (number->string i)) "DP.source" (string-append s_name (number->string i)))
    )
)

; Source Extension Doping
(sdedr:define-constant-profile "DP.sourceExt" SD_dopant SD_extension_doping)
(do ((i 0 (+ i 1)))
    ((= i num_of_fin))
    (begin
        (sdedr:define-constant-profile-region (string-append "DDP_Sext_" (number->string i)) "DP.sourceExt" (string-append sext_name (number->string i)))
    )
)

; Source Channel Doping
(sdedr:define-constant-profile "DP.channel" substrate_dopant C_doping)
(do ((i 0 (+ i 1)))
    ((= i num_of_fin))
    (begin
        (sdedr:define-constant-profile-region (string-append "DDP_C_" (number->string i)) "DP.channel" (string-append c_name (number->string i)))
    )
)

; Drain Doping
(sdedr:define-constant-profile "DP.drain" SD_dopant SD_doping)
(do ((i 0 (+ i 1)))
    ((= i num_of_fin))
    (begin
        (sdedr:define-constant-profile-region (string-append "DDP_D_" (number->string i)) "DP.drain" (string-append d_name (number->string i)))
    )
)

; Drain Extension Doping
(sdedr:define-constant-profile "DP.drainExt" SD_dopant SD_extension_doping)
(do ((i 0 (+ i 1)))
    ((= i num_of_fin))
    (begin
        (sdedr:define-constant-profile-region (string-append "DDP_Dext_" (number->string i)) "DP.drainExt" (string-append dext_name (number->string i)))
    )
)

; Substrate Doping
(sdedr:define-constant-profile "DP.Body" substrate_dopant Sub_Doping)
(sdedr:define-constant-profile-region "DPP.Body" "DP.Body" "R.Body" )

; (5). Mesh
; Globel mesh
(define gf_max 4)
(define gf_min 8)
(define all_list (get-body-list))

(sdedr:define-refeval-window
    "Rwin.global" 
    "cuboid"
    (position (sde:max-x all_list) (sde:max-y all_list) (sde:max-z all_list))
    (position (sde:min-x all_list) (sde:min-y all_list) (sde:min-z all_list))
)

(sdedr:define-refinement-size "RD.global"
    (/ (+ Lg (* 2 Lext)) gf_max) (/ y1 gf_max) (/ Hsub gf_max)
    (/ (+ Lg (* 2 Lext)) gf_min) (/ y1 gf_min) (/ Hsub gf_min)
)

(sdedr:define-refinement-function 
    "RD.global"
    "MaxLenInt" 
    "Silicon"
    "SiO2"
    5e-3
    1.5
)

(sdedr:define-refinement-function 
    "RD.global"
    "MaxLenInt" 
    "Silicon"
    "InterfacialOxide"
    1e-3
    1.5
    "DoubleSide"
)

(sdedr:define-refinement-placement "RP.global" "RD.global" "Rwin.global")

; Channel Mesh
(define chf_max 4)
(define chf_min 8)
(do ((i 0 (+ i 1)))
    ((= i num_of_fin))
    (begin

        (sdedr:define-refinement-size (string-append "RD.channel_" (number->string i))
            (/ Lg chf_max) (/ y1 chf_max) (/ Fh chf_max) 
            (/ Lg chf_min) (/ y1 chf_min) (/ Fh chf_min) 
        )

        (sdedr:define-refinement-region 
            (string-append "RP.channel_" (number->string i)) 
            (string-append "RD.channel_" (number->string i)) 
            (string-append c_name (number->string i))
        )

    )
)

; (6). Save
(sde:build-mesh "sde_result")