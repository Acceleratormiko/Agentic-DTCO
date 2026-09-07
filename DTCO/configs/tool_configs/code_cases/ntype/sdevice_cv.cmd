* (1) Define the device
Device "nMOS" {

  * (1.1) Input and output file for DC analysis
  File{
    Grid      = "sde_result_msh.tdr"
    Plot      = "DC_des.tdr"
    Parameter = "sdevice.par"
    Current   = "DC_des.plt"
  }
  
  * (1.2) Boundary conditionsd
  Electrode{
    { Name="source"    Voltage=0.0 Resist= 750 }
    { Name="drain"     Voltage=0.0 Resist= 750 }
    { Name="gate"      Voltage=0.0 }
    { Name="substrate" Voltage=0.0 }
  }

  * (1.3) Physics models
  Physics{
    Fermi
    eQuantumPotential
    EffectiveIntrinsicDensity( BandGapNarrowing(OldSlotboom) )     
    Mobility(
      DopingDep(BALMob(Lch=20.0))
      ThinLayer(IALMob)
      eHighFieldsaturation( GradQuasiFermi )
      hHighFieldsaturation( GradQuasiFermi )
      Enormal
    )
    Recombination(
      SRH( DopingDep TempDependence )
      * Phase 2: Add Band-to-band tunneling model to correct the off-current
      * SRH(DopingDep TempDependence Band2Band(Model = Hurkx))
    )           
  }
} * End of Device{}

* (2) Math
Math {
  Number_Of_Threads=4
  Extrapolate              
  Derivatives
  RelErrControl
  Digits=5                
  ErrRef(electron)=1.e10   
  ErrRef(hole)=1.e10        
  Iterations=40
  Notdamped=100
  Method = ParDiSo
}

* (3) Plot results
Plot{
  *--Density and Currents, etc
  eDensity hDensity
  TotalCurrent/Vector eCurrent/Vector hCurrent/Vector
  eMobility/Element hMobility/Element
  eVelocity hVelocity
  eQuasiFermi hQuasiFermi
  
  *--Temperature 
  eTemperature hTemperature Temperature
  
  *--Fields and charges
  ElectricField/Vector Potential SpaceCharge
  
  *--Doping Profiles
  Doping DonorConcentration AcceptorConcentration
  
  *--Generation/Recombination
  SRH Band2Band Auger
  ImpactIonization eImpactIonization hImpactIonization
  
  *--Driving forces
  eGradQuasiFermi/Vector hGradQuasiFermi/Vector
  eEparallel hEparallel eENormal hENormal
  
  *--Band structure/Composition
  BandGap 
  BandGapNarrowing
  Affinity
  ConductionBand ValenceBand
  eQuantumPotential hQuantumPotential
}

* (4) Output file for AC analysis
File {
  Output    = "CV_des.log"
  ACExtract = "_ac_des.plt"
}

* (5) Define the system
System {
  *-Physical devices:
  nMOS nmos1 ( "source"=s  "drain"=d "gate"=g "substrate"=b )
  
  *-Lumped elements:
  Vsource_pset vs (s 0) { dc = 0.0 }
  Vsource_pset vg (g 0) { dc = 0.0 }   
  Vsource_pset vb (b 0) { dc = 0.0 }  
  Vsource_pset vd (d 0) { dc = 0.0 }
}

* (6) Solve
Solve {
  Coupled(Iterations=100){ Poisson eQuantumPotential}
  Coupled{ Poisson Electron Hole eQuantumPotential }

  Quasistationary(
      InitialStep=0.01 Increment= 1.2 MinStep=1e-5 MaxStep=0.1
      Goal{ parameter=vg.dc Voltage= -0.75  }
  ) { Coupled { Poisson Electron Hole } }

  NewCurrentPrefix="result"
  Quasistationary(
      InitialStep=0.01 Increment= 1.2 MinStep=1e-5 MaxStep=0.1
      Goal{ parameter=vg.dc Voltage= 0.75 }
  ) { ACCoupled (
      StartFrequency=1e6 EndFrequency=1e6 NumberOfPoints=1 Decade
      Node(s d g b) Exclude(vs vd vg vb) 
      ACCompute (Time = (Range = (0 1)  Intervals = 30))
  ) { Poisson Electron Hole}
  }
}