# from the LLC90 run
# needs to be unicode for python 2/3 reasons
# io.StringIO needs a unicode string 
diagnostics=u""" Total Nb of available Diagnostics: ndiagt=   296
------------------------------------------------------------------------------------
  Num  |<-Name->|Levs|  mate |<- code ->|<--  Units   -->|<- Tile (max=80c)
------------------------------------------------------------------------------------
     1 |SDIAG1  |  1 |       |SM      L1|user-defined    |User-Defined   Surface   Diagnostic  #1
     2 |SDIAG2  |  1 |       |SM      L1|user-defined    |User-Defined   Surface   Diagnostic  #2
     3 |SDIAG3  |  1 |       |SM      L1|user-defined    |User-Defined   Surface   Diagnostic  #3
     4 |SDIAG4  |  1 |       |SM      L1|user-defined    |User-Defined   Surface   Diagnostic  #4
     5 |SDIAG5  |  1 |       |SM      L1|user-defined    |User-Defined   Surface   Diagnostic  #5
     6 |SDIAG6  |  1 |       |SM      L1|user-defined    |User-Defined   Surface   Diagnostic  #6
     7 |SDIAG7  |  1 |       |SU      L1|user-defined    |User-Defined U.pt Surface Diagnostic #7
     8 |SDIAG8  |  1 |       |SV      L1|user-defined    |User-Defined V.pt Surface Diagnostic #8
     9 |SDIAG9  |  1 |    10 |UU      L1|user-defined    |User-Defined U.vector Surface Diag.  #9
    10 |SDIAG10 |  1 |     9 |VV      L1|user-defined    |User-Defined V.vector Surface Diag. #10
    11 |UDIAG1  | 50 |       |SM      MR|user-defined    |User-Defined Model-Level Diagnostic  #1
    12 |UDIAG2  | 50 |       |SM      MR|user-defined    |User-Defined Model-Level Diagnostic  #2
    13 |UDIAG3  | 50 |       |SMR     MR|user-defined    |User-Defined Model-Level Diagnostic  #3
    14 |UDIAG4  | 50 |       |SMR     MR|user-defined    |User-Defined Model-Level Diagnostic  #4
    15 |UDIAG5  | 50 |       |SU      MR|user-defined    |User-Defined U.pt Model-Level Diag.  #5
    16 |UDIAG6  | 50 |       |SV      MR|user-defined    |User-Defined V.pt Model-Level Diag.  #6
    17 |UDIAG7  | 50 |    18 |UUR     MR|user-defined    |User-Defined U.vector Model-Lev Diag.#7
    18 |UDIAG8  | 50 |    17 |VVR     MR|user-defined    |User-Defined V.vector Model-Lev Diag.#8
    19 |UDIAG9  | 50 |       |SM      ML|user-defined    |User-Defined Phys-Level  Diagnostic  #9
    20 |UDIAG10 | 50 |       |SM      ML|user-defined    |User-Defined Phys-Level  Diagnostic #10
    21 |SDIAGC  |  1 |    22 |SM  C   L1|user-defined    |User-Defined Counted Surface Diagnostic
    22 |SDIAGCC |  1 |       |SM      L1|count           |User-Defined Surface Diagnostic Counter
    23 |ETAN    |  1 |       |SM      M1|m               |Surface Height Anomaly
    24 |ETANSQ  |  1 |       |SM P    M1|m^2             |Square of Surface Height Anomaly
    25 |DETADT2 |  1 |       |SM      M1|m^2/s^2         |Square of Surface Height Anomaly Tendency
    26 |THETA   | 50 |       |SMR     MR|degC            |Potential Temperature
    27 |SALT    | 50 |       |SMR     MR|psu             |Salinity
    28 |RELHUM  | 50 |       |SMR     MR|percent         |Relative Humidity
    29 |SALTanom| 50 |       |SMR     MR|psu             |Salt anomaly (=SALT-35; g/kg)
    30 |UVEL    | 50 |    31 |UUR     MR|m/s             |Zonal Component of Velocity (m/s)
    31 |VVEL    | 50 |    30 |VVR     MR|m/s             |Meridional Component of Velocity (m/s)
    32 |WVEL    | 50 |       |WM      LR|m/s             |Vertical Component of Velocity (r_units/s)
    33 |THETASQ | 50 |       |SMRP    MR|degC^2          |Square of Potential Temperature
    34 |SALTSQ  | 50 |       |SMRP    MR|(psu)^2         |Square of Salinity
    35 |SALTSQan| 50 |       |SMRP    MR|(psu)^2         |Square of Salt anomaly (=(SALT-35)^2 (g^2/kg^2)
    36 |UVELSQ  | 50 |    37 |UURP    MR|m^2/s^2         |Square of Zonal Comp of Velocity (m^2/s^2)
    37 |VVELSQ  | 50 |    36 |VVRP    MR|m^2/s^2         |Square of Meridional Comp of Velocity (m^2/s^2)
    38 |WVELSQ  | 50 |       |WM P    LR|m^2/s^2         |Square of Vertical Comp of Velocity
    39 |UE_VEL_C| 50 |    40 |UMR     MR|m/s             |Eastward Velocity (m/s) (cell center)
    40 |VN_VEL_C| 50 |    39 |VMR     MR|m/s             |Northward Velocity (m/s) (cell center)
    41 |UV_VEL_C| 50 |    41 |UMR     MR|m^2/s^2         |Product of horizontal Comp of velocity (cell center)
    42 |UV_VEL_Z| 50 |    42 |UZR     MR|m^2/s^2         |Meridional Transport of Zonal Momentum (m^2/s^2)
    43 |WU_VEL  | 50 |       |WU      LR|m.m/s^2         |Vertical Transport of Zonal Momentum
    44 |WV_VEL  | 50 |       |WV      LR|m.m/s^2         |Vertical Transport of Meridional Momentum
    45 |UVELMASS| 50 |    46 |UUr     MR|m/s             |Zonal Mass-Weighted Comp of Velocity (m/s)
    46 |VVELMASS| 50 |    45 |VVr     MR|m/s             |Meridional Mass-Weighted Comp of Velocity (m/s)
    47 |WVELMASS| 50 |       |WM      LR|m/s             |Vertical Mass-Weighted Comp of Velocity
    48 |PhiVEL  | 50 |    45 |SMR P   MR|m^2/s           |Horizontal Velocity Potential (m^2/s)
    49 |PsiVEL  | 50 |    48 |SZ  P   MR|m.m^2/s         |Horizontal Velocity Stream-Function
    50 |UTHMASS | 50 |    51 |UUr     MR|degC.m/s        |Zonal Mass-Weight Transp of Pot Temp
    51 |VTHMASS | 50 |    50 |VVr     MR|degC.m/s        |Meridional Mass-Weight Transp of Pot Temp
    52 |WTHMASS | 50 |       |WM      LR|degC.m/s        |Vertical Mass-Weight Transp of Pot Temp (K.m/s)
    53 |USLTMASS| 50 |    54 |UUr     MR|psu.m/s         |Zonal Mass-Weight Transp of Salinity
    54 |VSLTMASS| 50 |    53 |VVr     MR|psu.m/s         |Meridional Mass-Weight Transp of Salinity
    55 |WSLTMASS| 50 |       |WM      LR|psu.m/s         |Vertical Mass-Weight Transp of Salinity
    56 |UVELTH  | 50 |    57 |UUR     MR|degC.m/s        |Zonal Transport of Pot Temp
    57 |VVELTH  | 50 |    56 |VVR     MR|degC.m/s        |Meridional Transport of Pot Temp
    58 |WVELTH  | 50 |       |WM      LR|degC.m/s        |Vertical Transport of Pot Temp
    59 |UVELSLT | 50 |    60 |UUR     MR|psu.m/s         |Zonal Transport of Salinity
    60 |VVELSLT | 50 |    59 |VVR     MR|psu.m/s         |Meridional Transport of Salinity
    61 |WVELSLT | 50 |       |WM      LR|psu.m/s         |Vertical Transport of Salinity
    62 |UVELPHI | 50 |    63 |UUr     MR|m^3/s^3         |Zonal Mass-Weight Transp of Pressure Pot.(p/rho) Anomaly
    63 |VVELPHI | 50 |    62 |VVr     MR|m^3/s^3         |Merid. Mass-Weight Transp of Pressure Pot.(p/rho) Anomaly
    64 |RHOAnoma| 50 |       |SMR     MR|kg/m^3          |Density Anomaly (=Rho-rhoConst)
    65 |RHOANOSQ| 50 |       |SMRP    MR|kg^2/m^6        |Square of Density Anomaly (=(Rho-rhoConst)^2)
    66 |URHOMASS| 50 |    67 |UUr     MR|kg/m^2/s        |Zonal Transport of Density
    67 |VRHOMASS| 50 |    66 |VVr     MR|kg/m^2/s        |Meridional Transport of Density
    68 |WRHOMASS| 50 |       |WM      LR|kg/m^2/s        |Vertical Transport of Density
    69 |WdRHO_P | 50 |       |WM      LR|kg/m^2/s        |Vertical velocity times delta^k(Rho)_at-const-P
    70 |WdRHOdP | 50 |       |WM      LR|kg/m^2/s        |Vertical velocity times delta^k(Rho)_at-const-T,S
    71 |PHIHYD  | 50 |       |SMR     MR|m^2/s^2         |Hydrostatic Pressure Pot.(p/rho) Anomaly
    72 |PHIHYDSQ| 50 |       |SMRP    MR|m^4/s^4         |Square of Hyd. Pressure Pot.(p/rho) Anomaly
    73 |PHIBOT  |  1 |       |SM      M1|m^2/s^2         |Bottom Pressure Pot.(p/rho) Anomaly
    74 |PHIBOTSQ|  1 |       |SM P    M1|m^4/s^4         |Square of Bottom Pressure Pot.(p/rho) Anomaly
    75 |PHI_SURF|  1 |       |SM      M1|m^2/s^2         |Surface Dynamical Pressure Pot.(p/rho)
    76 |PHIHYDcR| 50 |       |SMR     MR|m^2/s^2         |Hydrostatic Pressure Pot.(p/rho) Anomaly @ const r
    77 |MXLDEPTH|  1 |       |SM      M1|m               |Mixed-Layer Depth (>0)
    78 |DRHODR  | 50 |       |SM      LR|kg/m^4          |Stratification: d.Sigma/dr (kg/m3/r_unit)
    79 |CONVADJ | 50 |       |SMR     LR|fraction        |Convective Adjustment Index [0-1]
    80 |oceTAUX |  1 |    81 |UU      U1|N/m^2           |zonal surface wind stress, >0 increases uVel
    81 |oceTAUY |  1 |    80 |VV      U1|N/m^2           |meridional surf. wind stress, >0 increases vVel
    82 |atmPload|  1 |       |SM      U1|Pa              |Atmospheric pressure loading
    83 |sIceLoad|  1 |       |SM      U1|kg/m^2          |sea-ice loading (in Mass of ice+snow / area unit)
    84 |oceFWflx|  1 |       |SM      U1|kg/m^2/s        |net surface Fresh-Water flux into the ocean (+=down), >0 decreases salinity
    85 |oceSflux|  1 |       |SM      U1|g/m^2/s         |net surface Salt flux into the ocean (+=down), >0 increases salinity
    86 |oceQnet |  1 |       |SM      U1|W/m^2           |net surface heat flux into the ocean (+=down), >0 increases theta
    87 |oceQsw  |  1 |       |SM      U1|W/m^2           |net Short-Wave radiation (+=down), >0 increases theta
    88 |oceFreez|  1 |       |SM      U1|W/m^2           |heating from freezing of sea-water (allowFreezing=T)
    89 |TRELAX  |  1 |       |SM      U1|W/m^2           |surface temperature relaxation, >0 increases theta
    90 |SRELAX  |  1 |       |SM      U1|g/m^2/s         |surface salinity relaxation, >0 increases salt
    91 |surForcT|  1 |       |SM      U1|W/m^2           |model surface forcing for Temperature, >0 increases theta
    92 |surForcS|  1 |       |SM      U1|g/m^2/s         |model surface forcing for Salinity, >0 increases salinity
    93 |TFLUX   |  1 |       |SM      U1|W/m^2           |total heat flux (match heat-content variations), >0 increases theta
    94 |SFLUX   |  1 |       |SM      U1|g/m^2/s         |total salt flux (match salt-content variations), >0 increases salt
    95 |RCENTER | 50 |       |SM      MR|m               |Cell-Center Height
    96 |RSURF   |  1 |       |SM      M1|m               |Surface Height
    97 |TOTUTEND| 50 |    98 |UUR     MR|m/s/day         |Tendency of Zonal Component of Velocity
    98 |TOTVTEND| 50 |    97 |VVR     MR|m/s/day         |Tendency of Meridional Component of Velocity
    99 |TOTTTEND| 50 |       |SMR     MR|degC/day        |Tendency of Potential Temperature
------------------------------------------------------------------------------------
  Num  |<-Name->|Levs|  mate |<- code ->|<--  Units   -->|<- Tile (max=80c)
------------------------------------------------------------------------------------
   100 |TOTSTEND| 50 |       |SMR     MR|psu/day         |Tendency of Salinity
   101 |MoistCor| 50 |       |SM      MR|W/m^2           |Heating correction due to moist thermodynamics
   102 |gT_Forc | 50 |       |SMR     MR|degC/s          |Potential Temp. forcing tendency
   103 |gS_Forc | 50 |       |SMR     MR|psu/s           |Salinity forcing tendency
   104 |AB_gT   | 50 |       |SMR     MR|degC/s          |Potential Temp. tendency from Adams-Bashforth
   105 |AB_gS   | 50 |       |SMR     MR|psu/s           |Salinity tendency from Adams-Bashforth
   106 |gTinAB  | 50 |       |SMR     MR|degC/s          |Potential Temp. tendency going in Adams-Bashforth
   107 |gSinAB  | 50 |       |SMR     MR|psu/s           |Salinity tendency going in Adams-Bashforth
   108 |AB_gU   | 50 |   109 |UUR     MR|m/s^2           |U momentum tendency from Adams-Bashforth
   109 |AB_gV   | 50 |   108 |VVR     MR|m/s^2           |V momentum tendency from Adams-Bashforth
   110 |ADVr_TH | 50 |       |WM      LR|degC.m^3/s      |Vertical   Advective Flux of Pot.Temperature
   111 |ADVx_TH | 50 |   112 |UU      MR|degC.m^3/s      |Zonal      Advective Flux of Pot.Temperature
   112 |ADVy_TH | 50 |   111 |VV      MR|degC.m^3/s      |Meridional Advective Flux of Pot.Temperature
   113 |DFrE_TH | 50 |       |WM      LR|degC.m^3/s      |Vertical Diffusive Flux of Pot.Temperature (Explicit part)
   114 |DFxE_TH | 50 |   115 |UU      MR|degC.m^3/s      |Zonal      Diffusive Flux of Pot.Temperature
   115 |DFyE_TH | 50 |   114 |VV      MR|degC.m^3/s      |Meridional Diffusive Flux of Pot.Temperature
   116 |DFrI_TH | 50 |       |WM      LR|degC.m^3/s      |Vertical Diffusive Flux of Pot.Temperature (Implicit part)
   117 |ADVr_SLT| 50 |       |WM      LR|psu.m^3/s       |Vertical   Advective Flux of Salinity
   118 |ADVx_SLT| 50 |   119 |UU      MR|psu.m^3/s       |Zonal      Advective Flux of Salinity
   119 |ADVy_SLT| 50 |   118 |VV      MR|psu.m^3/s       |Meridional Advective Flux of Salinity
   120 |DFrE_SLT| 50 |       |WM      LR|psu.m^3/s       |Vertical Diffusive Flux of Salinity    (Explicit part)
   121 |DFxE_SLT| 50 |   122 |UU      MR|psu.m^3/s       |Zonal      Diffusive Flux of Salinity
   122 |DFyE_SLT| 50 |   121 |VV      MR|psu.m^3/s       |Meridional Diffusive Flux of Salinity
   123 |DFrI_SLT| 50 |       |WM      LR|psu.m^3/s       |Vertical Diffusive Flux of Salinity    (Implicit part)
   124 |SALTFILL| 50 |       |SM      MR|psu.m^3/s       |Filling of Negative Values of Salinity
   125 |VISCAHZ | 50 |       |SZ      MR|m^2/s           |Harmonic Visc Coefficient (m2/s) (Zeta Pt)
   126 |VISCA4Z | 50 |       |SZ      MR|m^4/s           |Biharmonic Visc Coefficient (m4/s) (Zeta Pt)
   127 |VISCAHD | 50 |       |SM      MR|m^2/s           |Harmonic Viscosity Coefficient (m2/s) (Div Pt)
   128 |VISCA4D | 50 |       |SM      MR|m^4/s           |Biharmonic Viscosity Coefficient (m4/s) (Div Pt)
   129 |VISCAHW | 50 |       |WM      LR|m^2/s           |Harmonic Viscosity Coefficient (m2/s) (W Pt)
   130 |VISCA4W | 50 |       |WM      LR|m^4/s           |Biharmonic Viscosity Coefficient (m4/s) (W Pt)
   131 |VAHZMAX | 50 |       |SZ      MR|m^2/s           |CFL-MAX Harm Visc Coefficient (m2/s) (Zeta Pt)
   132 |VA4ZMAX | 50 |       |SZ      MR|m^4/s           |CFL-MAX Biharm Visc Coefficient (m4/s) (Zeta Pt)
   133 |VAHDMAX | 50 |       |SM      MR|m^2/s           |CFL-MAX Harm Visc Coefficient (m2/s) (Div Pt)
   134 |VA4DMAX | 50 |       |SM      MR|m^4/s           |CFL-MAX Biharm Visc Coefficient (m4/s) (Div Pt)
   135 |VAHZMIN | 50 |       |SZ      MR|m^2/s           |RE-MIN Harm Visc Coefficient (m2/s) (Zeta Pt)
   136 |VA4ZMIN | 50 |       |SZ      MR|m^4/s           |RE-MIN Biharm Visc Coefficient (m4/s) (Zeta Pt)
   137 |VAHDMIN | 50 |       |SM      MR|m^2/s           |RE-MIN Harm Visc Coefficient (m2/s) (Div Pt)
   138 |VA4DMIN | 50 |       |SM      MR|m^4/s           |RE-MIN Biharm Visc Coefficient (m4/s) (Div Pt)
   139 |VAHZLTH | 50 |       |SZ      MR|m^2/s           |Leith Harm Visc Coefficient (m2/s) (Zeta Pt)
   140 |VA4ZLTH | 50 |       |SZ      MR|m^4/s           |Leith Biharm Visc Coefficient (m4/s) (Zeta Pt)
   141 |VAHDLTH | 50 |       |SM      MR|m^2/s           |Leith Harm Visc Coefficient (m2/s) (Div Pt)
   142 |VA4DLTH | 50 |       |SM      MR|m^4/s           |Leith Biharm Visc Coefficient (m4/s) (Div Pt)
   143 |VAHZLTHD| 50 |       |SZ      MR|m^2/s           |LeithD Harm Visc Coefficient (m2/s) (Zeta Pt)
   144 |VA4ZLTHD| 50 |       |SZ      MR|m^4/s           |LeithD Biharm Visc Coefficient (m4/s) (Zeta Pt)
   145 |VAHDLTHD| 50 |       |SM      MR|m^2/s           |LeithD Harm Visc Coefficient (m2/s) (Div Pt)
   146 |VA4DLTHD| 50 |       |SM      MR|m^4/s           |LeithD Biharm Visc Coefficient (m4/s) (Div Pt)
   147 |VAHZSMAG| 50 |       |SZ      MR|m^2/s           |Smagorinsky Harm Visc Coefficient (m2/s) (Zeta Pt)
   148 |VA4ZSMAG| 50 |       |SZ      MR|m^4/s           |Smagorinsky Biharm Visc Coeff. (m4/s) (Zeta Pt)
   149 |VAHDSMAG| 50 |       |SM      MR|m^2/s           |Smagorinsky Harm Visc Coefficient (m2/s) (Div Pt)
   150 |VA4DSMAG| 50 |       |SM      MR|m^4/s           |Smagorinsky Biharm Visc Coeff. (m4/s) (Div Pt)
   151 |momKE   | 50 |       |SMR     MR|m^2/s^2         |Kinetic Energy (in momentum Eq.)
   152 |momHDiv | 50 |       |SMR     MR|s^-1            |Horizontal Divergence (in momentum Eq.)
   153 |momVort3| 50 |       |SZR     MR|s^-1            |3rd component (vertical) of Vorticity
   154 |Strain  | 50 |       |SZR     MR|s^-1            |Horizontal Strain of Horizontal Velocities
   155 |Tension | 50 |       |SMR     MR|s^-1            |Horizontal Tension of Horizontal Velocities
   156 |UBotDrag| 50 |   157 |UUR     MR|m/s^2           |U momentum tendency from Bottom Drag
   157 |VBotDrag| 50 |   156 |VVR     MR|m/s^2           |V momentum tendency from Bottom Drag
   158 |USidDrag| 50 |   159 |UUR     MR|m/s^2           |U momentum tendency from Side Drag
   159 |VSidDrag| 50 |   158 |VVR     MR|m/s^2           |V momentum tendency from Side Drag
   160 |Um_Diss | 50 |   161 |UUR     MR|m/s^2           |U momentum tendency from Dissipation
   161 |Vm_Diss | 50 |   160 |VVR     MR|m/s^2           |V momentum tendency from Dissipation
   162 |Um_Advec| 50 |   163 |UUR     MR|m/s^2           |U momentum tendency from Advection terms
   163 |Vm_Advec| 50 |   162 |VVR     MR|m/s^2           |V momentum tendency from Advection terms
   164 |Um_Cori | 50 |   165 |UUR     MR|m/s^2           |U momentum tendency from Coriolis term
   165 |Vm_Cori | 50 |   164 |VVR     MR|m/s^2           |V momentum tendency from Coriolis term
   166 |Um_dPHdx| 50 |   167 |UUR     MR|m/s^2           |U momentum tendency from Hydrostatic Pressure grad
   167 |Vm_dPHdy| 50 |   166 |VVR     MR|m/s^2           |V momentum tendency from Hydrostatic Pressure grad
   168 |Um_Ext  | 50 |   169 |UUR     MR|m/s^2           |U momentum tendency from external forcing
   169 |Vm_Ext  | 50 |   168 |VVR     MR|m/s^2           |V momentum tendency from external forcing
   170 |Um_AdvZ3| 50 |   171 |UUR     MR|m/s^2           |U momentum tendency from Vorticity Advection
   171 |Vm_AdvZ3| 50 |   170 |VVR     MR|m/s^2           |V momentum tendency from Vorticity Advection
   172 |Um_AdvRe| 50 |   173 |UUR     MR|m/s^2           |U momentum tendency from vertical Advection (Explicit part)
   173 |Vm_AdvRe| 50 |   172 |VVR     MR|m/s^2           |V momentum tendency from vertical Advection (Explicit part)
   174 |VISrI_Um| 50 |       |WU      LR|m^4/s^2         |Vertical   Viscous Flux of U momentum (Implicit part)
   175 |VISrI_Vm| 50 |       |WV      LR|m^4/s^2         |Vertical   Viscous Flux of V momentum (Implicit part)
   176 |EXFhs   |  1 |       |SM      U1|W/m^2           |Sensible heat flux into ocean, >0 increases theta
   177 |EXFhl   |  1 |       |SM      U1|W/m^2           |Latent heat flux into ocean, >0 increases theta
   178 |EXFlwnet|  1 |       |SM      U1|W/m^2           |Net upward longwave radiation, >0 decreases theta
   179 |EXFswnet|  1 |       |SM      U1|W/m^2           |Net upward shortwave radiation, >0 decreases theta
   180 |EXFlwdn |  1 |       |SM      U1|W/m^2           |Downward longwave radiation, >0 increases theta
   181 |EXFswdn |  1 |       |SM      U1|W/m^2           |Downward shortwave radiation, >0 increases theta
   182 |EXFqnet |  1 |       |SM      U1|W/m^2           |Net upward heat flux (turb+rad), >0 decreases theta
   183 |EXFtaux |  1 |       |UM      U1|N/m^2           |zonal surface wind stress, >0 increases uVel
   184 |EXFtauy |  1 |       |VM      U1|N/m^2           |meridional surface wind stress, >0 increases vVel
   185 |EXFuwind|  1 |       |UM      U1|m/s             |zonal 10-m wind speed, >0 increases uVel
   186 |EXFvwind|  1 |       |VM      U1|m/s             |meridional 10-m wind speed, >0 increases uVel
   187 |EXFwspee|  1 |       |SM      U1|m/s             |10-m wind speed modulus ( >= 0 )
   188 |EXFatemp|  1 |       |SM      U1|degK            |surface (2-m) air temperature
   189 |EXFaqh  |  1 |       |SM      U1|kg/kg           |surface (2-m) specific humidity
   190 |EXFevap |  1 |       |SM      U1|m/s             |evaporation, > 0 increases salinity
   191 |EXFpreci|  1 |       |SM      U1|m/s             |precipitation, > 0 decreases salinity
   192 |EXFsnow |  1 |       |SM      U1|m/s             |snow precipitation, > 0 decreases salinity
   193 |EXFempmr|  1 |       |SM      U1|m/s             |net upward freshwater flux, > 0 increases salinity
   194 |EXFpress|  1 |       |SM      U1|N/m^2           |atmospheric pressure field
   195 |EXFroff |  1 |       |SM      U1|m/s             |river runoff, > 0 decreases salinity
   196 |EXFroft |  1 |       |SM      U1|deg C           |river runoff temperature
   197 |GGL90TKE| 50 |       |SM      LR|m^2/s^2         |GGL90 sub-grid turbulent kinetic energy
   198 |GGL90Lmx| 50 |       |SM      LR|m               |Mixing length scale
   199 |GGL90Prl| 50 |       |SM      LR|1               |Prandtl number used in GGL90
------------------------------------------------------------------------------------
  Num  |<-Name->|Levs|  mate |<- code ->|<--  Units   -->|<- Tile (max=80c)
------------------------------------------------------------------------------------
   200 |GGL90ArU| 50 |       |SM      LR|m^2/s           |GGL90 eddy viscosity at U-point
   201 |GGL90ArV| 50 |       |SM      LR|m^2/s           |GGL90 eddy viscosity at V-point
   202 |GGL90Kr | 50 |       |SM      LR|m^2/s           |GGL90 diffusion coefficient for temperature
   203 |GGL90flx|  1 |       |SM      L1|m^3/s^3         |Surface flux of TKE
   204 |GGL90tau|  1 |       |SM      L1|m^3/s^3         |Work done by the wind
   205 |GM_VisbK|  1 |       |SM P    M1|m^2/s           |Mixing coefficient from Visbeck etal parameterization
   206 |GM_hTrsL|  1 |       |SM P    M1|m               |Base depth (>0) of the Transition Layer
   207 |GM_baseS|  1 |       |SM P    M1|1               |Slope at the base of the Transition Layer
   208 |GM_rLamb|  1 |       |SM P    M1|1/m             |Slope vertical gradient at Trans. Layer Base (=recip.Lambda)
   209 |GM_Kux  | 50 |   210 |UU P    MR|m^2/s           |K_11 element (U.point, X.dir) of GM-Redi tensor
   210 |GM_Kvy  | 50 |   209 |VV P    MR|m^2/s           |K_22 element (V.point, Y.dir) of GM-Redi tensor
   211 |GM_Kuz  | 50 |   212 |UU      MR|m^2/s           |K_13 element (U.point, Z.dir) of GM-Redi tensor
   212 |GM_Kvz  | 50 |   211 |VV      MR|m^2/s           |K_23 element (V.point, Z.dir) of GM-Redi tensor
   213 |GM_Kwx  | 50 |   214 |UM      LR|m^2/s           |K_31 element (W.point, X.dir) of GM-Redi tensor
   214 |GM_Kwy  | 50 |   213 |VM      LR|m^2/s           |K_32 element (W.point, Y.dir) of GM-Redi tensor
   215 |GM_Kwz  | 50 |       |WM P    LR|m^2/s           |K_33 element (W.point, Z.dir) of GM-Redi tensor
   216 |GM_PsiX | 50 |   217 |UU      LR|m^2/s           |GM Bolus transport stream-function : U component
   217 |GM_PsiY | 50 |   216 |VV      LR|m^2/s           |GM Bolus transport stream-function : V component
   218 |GM_KuzTz| 50 |   219 |UU      MR|degC.m^3/s      |Redi Off-diagonal Temperature flux: X component
   219 |GM_KvzTz| 50 |   218 |VV      MR|degC.m^3/s      |Redi Off-diagonal Temperature flux: Y component
   220 |GM_KwzTz| 50 |       |WM      LR|degC.m^3/s      |Redi main-diagonal vertical Temperature flux
   221 |GM_ubT  | 50 |   222 |UUr     MR|degC.m^3/s      |Zonal Mass-Weight Bolus Transp of Pot Temp
   222 |GM_vbT  | 50 |   221 |VVr     MR|degC.m^3/s      |Meridional Mass-Weight Bolus Transp of Pot Temp
   223 |SIarea  |  1 |       |SM      M1|m^2/m^2         |SEAICE fractional ice-covered area [0 to 1]
   224 |SIareaPR|  1 |       |SM      M1|m^2/m^2         |SIarea preceeding ridging process
   225 |SIareaPT|  1 |       |SM      M1|m^2/m^2         |SIarea preceeding thermodynamic growth/melt
   226 |SIheff  |  1 |       |SM      M1|m               |SEAICE effective ice thickness
   227 |SIheffPT|  1 |       |SM      M1|m               |SIheff preceeeding thermodynamic growth/melt
   228 |SIhsnow |  1 |       |SM      M1|m               |SEAICE effective snow thickness
   229 |SIhsnoPT|  1 |       |SM      M1|m               |SIhsnow preceeeding thermodynamic growth/melt
   230 |SIhsalt |  1 |       |SM      M1|g/m^2           |SEAICE effective salinity
   231 |SItices |  1 |   223 |SM  C   M1|K               |Surface Temperature over Sea-Ice (area weighted)
   232 |SIuice  |  1 |   233 |UU      M1|m/s             |SEAICE zonal ice velocity, >0 from West to East
   233 |SIvice  |  1 |   232 |VV      M1|m/s             |SEAICE merid. ice velocity, >0 from South to North
   234 |SIfu    |  1 |   235 |UU      U1|N/m^2           |SEAICE zonal surface wind stress, >0 increases uVel
   235 |SIfv    |  1 |   234 |VV      U1|N/m^2           |SEAICE merid. surface wind stress, >0 increases vVel
   236 |SIuwind |  1 |   237 |UM      U1|m/s             |SEAICE zonal 10-m wind speed, >0 increases uVel
   237 |SIvwind |  1 |   236 |VM      U1|m/s             |SEAICE meridional 10-m wind speed, >0 increases uVel
   238 |SIqnet  |  1 |       |SM      U1|W/m^2           |Ocean surface heatflux, turb+rad, >0 decreases theta
   239 |SIqsw   |  1 |       |SM      U1|W/m^2           |Ocean surface shortwave radiat., >0 decreases theta
   240 |SIatmQnt|  1 |       |SM      U1|W/m^2           |Net atmospheric heat flux, >0 decreases theta
   241 |SItflux |  1 |       |SM      U1|W/m^2           |Same as TFLUX but incl seaice (>0 incr T decr H)
   242 |SIaaflux|  1 |       |SM      U1|W/m^2           |conservative ocn<->seaice adv. heat flux adjust.
   243 |SIhl    |  1 |       |SM      U1|W/m^2           |Latent heat flux into ocean, >0 increases theta
   244 |SIqneto |  1 |       |SM      U1|W/m^2           |Open Ocean Part of SIqnet, turb+rad, >0 decr theta
   245 |SIqneti |  1 |       |SM      U1|W/m^2           |Ice Covered Part of SIqnet, turb+rad, >0 decr theta
   246 |SIempmr |  1 |       |SM      U1|kg/m^2/s        |Ocean surface freshwater flux, > 0 increases salt
   247 |SIatmFW |  1 |       |SM      U1|kg/m^2/s        |Net freshwater flux from atmosphere & land (+=down)
   248 |SIsnPrcp|  1 |       |SM      U1|kg/m^2/s        |Snow precip. (+=dw) over Sea-Ice (area weighted)
   249 |SIfwSubl|  1 |       |SM      U1|kg/m^2/s        |Potential sublimation freshwater flux, >0 decr. ice
   250 |SIacSubl|  1 |       |SM      U1|kg/m^2/s        |Actual sublimation freshwater flux, >0 decr. ice
   251 |SIrsSubl|  1 |       |SM      U1|kg/m^2/s        |Residual subl. freshwater flux, >0 taken from ocn
   252 |SIactLHF|  1 |       |SM      U1|W/m^2           |Actual latent heat flux over ice
   253 |SImaxLHF|  1 |       |SM      U1|W/m^2           |Maximum latent heat flux over ice
   254 |SIaQbOCN|  1 |       |SM      M1|m/s             |Potential HEFF rate of change by ocean ice flux
   255 |SIaQbATC|  1 |       |SM      M1|m/s             |Potential HEFF rate of change by atm flux over ice
   256 |SIaQbATO|  1 |       |SM      M1|m/s             |Potential HEFF rate of change by open ocn atm flux
   257 |SIdHbOCN|  1 |       |SM      M1|m/s             |HEFF rate of change by ocean ice flux
   258 |SIdSbATC|  1 |       |SM      M1|m/s             |HSNOW rate of change by atm flux over sea ice
   259 |SIdSbOCN|  1 |       |SM      M1|m/s             |HSNOW rate of change by ocean ice flux
   260 |SIdHbATC|  1 |       |SM      M1|m/s             |HEFF rate of change by atm flux over sea ice
   261 |SIdHbATO|  1 |       |SM      M1|m/s             |HEFF rate of change by open ocn atm flux
   262 |SIdHbFLO|  1 |       |SM      M1|m/s             |HEFF rate of change by flooding snow
   263 |SIdAbATO|  1 |       |SM      M1|m^2/m^2/s       |Potential AREA rate of change by open ocn atm flux
   264 |SIdAbATC|  1 |       |SM      M1|m^2/m^2/s       |Potential AREA rate of change by atm flux over ice
   265 |SIdAbOCN|  1 |       |SM      M1|m^2/m^2/s       |Potential AREA rate of change by ocean ice flux
   266 |SIdA    |  1 |       |SM      M1|m^2/m^2/s       |AREA rate of change (net)
   267 |ADVxHEFF|  1 |   268 |UU      M1|m.m^2/s         |Zonal      Advective Flux of eff ice thickn
   268 |ADVyHEFF|  1 |   267 |VV      M1|m.m^2/s         |Meridional Advective Flux of eff ice thickn
   269 |SIuheff |  1 |   270 |UU      M1|m^2/s           |Zonal      Transport of eff ice thickn (centered)
   270 |SIvheff |  1 |   269 |VV      M1|m^2/s           |Meridional Transport of eff ice thickn (centered)
   271 |DFxEHEFF|  1 |   272 |UU      M1|m^2/s           |Zonal      Diffusive Flux of eff ice thickn
   272 |DFyEHEFF|  1 |   271 |VV      M1|m^2/s           |Meridional Diffusive Flux of eff ice thickn
   273 |ADVxAREA|  1 |   274 |UU      M1|m^2/m^2.m^2/s   |Zonal      Advective Flux of fract area
   274 |ADVyAREA|  1 |   273 |VV      M1|m^2/m^2.m^2/s   |Meridional Advective Flux of fract area
   275 |DFxEAREA|  1 |   276 |UU      M1|m^2/m^2.m^2/s   |Zonal      Diffusive Flux of fract area
   276 |DFyEAREA|  1 |   275 |VV      M1|m^2/m^2.m^2/s   |Meridional Diffusive Flux of fract area
   277 |ADVxSNOW|  1 |   278 |UU      M1|m.m^2/s         |Zonal      Advective Flux of eff snow thickn
   278 |ADVySNOW|  1 |   277 |VV      M1|m.m^2/s         |Meridional Advective Flux of eff snow thickn
   279 |DFxESNOW|  1 |   280 |UU      M1|m.m^2/s         |Zonal      Diffusive Flux of eff snow thickn
   280 |DFyESNOW|  1 |   279 |VV      M1|m.m^2/s         |Meridional Diffusive Flux of eff snow thickn
   281 |ADVxSSLT|  1 |   282 |UU      M1|psu.m^2/s       |Zonal      Advective Flux of seaice salinity
   282 |ADVySSLT|  1 |   281 |VV      M1|psu.m^2/s       |Meridional Advective Flux of seaice salinity
   283 |DFxESSLT|  1 |   284 |UU      M1|psu.m^2/s       |Zonal      Diffusive Flux of seaice salinity
   284 |DFyESSLT|  1 |   283 |VV      M1|psu.m^2/s       |Meridional Diffusive Flux of seaice salinity
   285 |SIpress |  1 |       |SM      M1|m^2/s^2         |SEAICE strength (with upper and lower limit)
   286 |SIzeta  |  1 |       |SM      M1|m^2/s           |SEAICE nonlinear bulk viscosity
   287 |SIeta   |  1 |       |SM      M1|m^2/s           |SEAICE nonlinear shear viscosity
   288 |SIsigI  |  1 |       |SM      M1|no units        |SEAICE normalized principle stress, component one
   289 |SIsigII |  1 |       |SM      M1|no units        |SEAICE normalized principle stress, component two
   290 |SIshear |  1 |       |SM      M1|s^{-1}          |SEAICE shear deformation
   291 |SIdelta |  1 |       |SM      M1|s^{-1}          |SEAICE Delta deformation
   292 |SItensil|  1 |       |SM      M1|m^2/s^2         |SEAICE maximal tensile strength
   293 |PLUMEKB | 50 |       |SM      MR|                |fractional plume: [0-1] (unitless)
   294 |oceSPtnd| 50 |       |SM      MR|g/m^2/s         |salt tendency due to salt plume flux >0 increases salinity
   295 |oceSPflx|  1 |       |SM      U1|g/m^2/s         |net surface Salt flux rejected into the ocean during freezing, (+=down),
   296 |oceSPDep|  1 |       |SM      U1|m               |Salt plume depth based on density criterion (>0)
------------------------------------------------------------------------------------
  Num  |<-Name->|Levs|  mate |<- code ->|<--  Units   -->|<- Tile (max=80c)
------------------------------------------------------------------------------------
"""
