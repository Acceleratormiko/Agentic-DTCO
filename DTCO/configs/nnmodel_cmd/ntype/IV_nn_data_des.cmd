* (1) Input and output file
File{
  Grid      = "sde_result_msh.tdr"
  Plot      = "IdVd_des.tdr"
  Parameter = "sdevice.par"
  Current   = "IdVd_des.plt"
  Output    = "IdVd_des.log"
}

* (2) Boundary conditions
Electrode{
  { Name="source"    Voltage=0.0 Resist= 750.0 }
  { Name="drain"     Voltage=0.0 Resist= 750.0 }
  { Name="gate"      Voltage=0.0 }
  { Name="substrate" Voltage=0.0 }
}

* (3) Physics models
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

* (4) Math
Math {
  Number_Of_Threads=4
  Extrapolate              
  Derivatives
  RelErrControl
  Digits=5                
  ErrRef(electron)=1.e10   
  ErrRef(hole)=1.e10        
  Iterations=30
  Notdamped=100
  Method = ParDiSo
}

* (5) Plot results
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

* (6) Solve
Solve {
  Coupled(Iterations=150){ Poisson eQuantumPotential}
  Coupled{ Poisson Electron Hole eQuantumPotential }

  Quasistationary(
      InitialStep=0.01 Increment= 1.2 MinStep=1e-5 MaxStep=0.1
      Goal{ Name="gate" Voltage= 0.0  }
  ) { Coupled { Poisson Electron Hole eQuantumPotential } }

  NewCurrentPrefix="result_Vg_0p0_"
  Quasistationary(
      InitialStep=0.01 Increment= 1.2 MinStep=1e-5 MaxStep=0.1
      Goal{ Name="drain" Voltage= 0.8  }
  ) { Coupled { Poisson Electron Hole eQuantumPotential } 
      CurrentPlot(Time=(Range=(0 1) Intervals=40))}

  NewCurrentPrefix="result_tmp_"
  Quasistationary(
      InitialStep=0.01 Increment= 1.2 MinStep=1e-5 MaxStep=0.1
      Goal{ Name="gate" Voltage= 0.1  }
  ) { Coupled { Poisson Electron Hole eQuantumPotential } }

  NewCurrentPrefix="result_Vg_0p1_"
  Quasistationary(
      InitialStep=0.01 Increment= 1.2 MinStep=1e-5 MaxStep=0.1
      Goal{ Name="drain" Voltage= 0.0  }
  ) { Coupled { Poisson Electron Hole eQuantumPotential } 
      CurrentPlot(Time=(Range=(0 1) Intervals=40))}

  NewCurrentPrefix="result_tmp_"
  Quasistationary(
      InitialStep=0.01 Increment= 1.2 MinStep=1e-5 MaxStep=0.1
      Goal{ Name="gate" Voltage= 0.2  }
  ) { Coupled { Poisson Electron Hole eQuantumPotential } }

  NewCurrentPrefix="result_Vg_0p2_"
  Quasistationary(
      InitialStep=0.01 Increment= 1.2 MinStep=1e-5 MaxStep=0.1
      Goal{ Name="drain" Voltage= 0.8  }
  ) { Coupled { Poisson Electron Hole eQuantumPotential } 
      CurrentPlot(Time=(Range=(0 1) Intervals=40))}

  NewCurrentPrefix="result_tmp_"
  Quasistationary(
      InitialStep=0.01 Increment= 1.2 MinStep=1e-5 MaxStep=0.1
      Goal{ Name="gate" Voltage= 0.3  }
  ) { Coupled { Poisson Electron Hole eQuantumPotential } }

  NewCurrentPrefix="result_Vg_0p3_"
  Quasistationary(
      InitialStep=0.01 Increment= 1.2 MinStep=1e-5 MaxStep=0.1
      Goal{ Name="drain" Voltage= 0.0  }
  ) { Coupled { Poisson Electron Hole eQuantumPotential } 
      CurrentPlot(Time=(Range=(0 1) Intervals=40))}

  NewCurrentPrefix="result_tmp_"
  Quasistationary(
      InitialStep=0.01 Increment= 1.2 MinStep=1e-5 MaxStep=0.1
      Goal{ Name="gate" Voltage= 0.4  }
  ) { Coupled { Poisson Electron Hole eQuantumPotential } }

  NewCurrentPrefix="result_Vg_0p4_"
  Quasistationary(
      InitialStep=0.01 Increment= 1.2 MinStep=1e-5 MaxStep=0.1
      Goal{ Name="drain" Voltage= 0.8  }
  ) { Coupled { Poisson Electron Hole eQuantumPotential } 
      CurrentPlot(Time=(Range=(0 1) Intervals=40))}

  NewCurrentPrefix="result_tmp_"
  Quasistationary(
      InitialStep=0.01 Increment= 1.2 MinStep=1e-5 MaxStep=0.1
      Goal{ Name="gate" Voltage= 0.5  }
  ) { Coupled { Poisson Electron Hole eQuantumPotential } }

  NewCurrentPrefix="result_Vg_0p5_"
  Quasistationary(
      InitialStep=0.01 Increment= 1.2 MinStep=1e-5 MaxStep=0.1
      Goal{ Name="drain" Voltage= 0.0  }
  ) { Coupled { Poisson Electron Hole eQuantumPotential } 
      CurrentPlot(Time=(Range=(0 1) Intervals=40))}

  NewCurrentPrefix="result_tmp_"
  Quasistationary(
      InitialStep=0.01 Increment= 1.2 MinStep=1e-5 MaxStep=0.1
      Goal{ Name="gate" Voltage= 0.6  }
  ) { Coupled { Poisson Electron Hole eQuantumPotential } }

  NewCurrentPrefix="result_Vg_0p6_"
  Quasistationary(
      InitialStep=0.01 Increment= 1.2 MinStep=1e-5 MaxStep=0.1
      Goal{ Name="drain" Voltage= 0.8  }
  ) { Coupled { Poisson Electron Hole eQuantumPotential } 
      CurrentPlot(Time=(Range=(0 1) Intervals=40))}

  NewCurrentPrefix="result_tmp_"
  Quasistationary(
      InitialStep=0.01 Increment= 1.2 MinStep=1e-5 MaxStep=0.1
      Goal{ Name="gate" Voltage= 0.7  }
  ) { Coupled { Poisson Electron Hole eQuantumPotential } }

  NewCurrentPrefix="result_Vg_0p7_"
  Quasistationary(
      InitialStep=0.01 Increment= 1.2 MinStep=1e-5 MaxStep=0.1
      Goal{ Name="drain" Voltage= 0.0  }
  ) { Coupled { Poisson Electron Hole eQuantumPotential } 
      CurrentPlot(Time=(Range=(0 1) Intervals=40))}

  NewCurrentPrefix="result_tmp_"
  Quasistationary(
      InitialStep=0.01 Increment= 1.2 MinStep=1e-5 MaxStep=0.1
      Goal{ Name="gate" Voltage= 0.8  }
  ) { Coupled { Poisson Electron Hole eQuantumPotential } }

  NewCurrentPrefix="result_Vg_0p8_"
  Quasistationary(
      InitialStep=0.01 Increment= 1.2 MinStep=1e-5 MaxStep=0.1
      Goal{ Name="drain" Voltage= 0.8  }
  ) { Coupled { Poisson Electron Hole eQuantumPotential } 
      CurrentPlot(Time=(Range=(0 1) Intervals=40))}

}