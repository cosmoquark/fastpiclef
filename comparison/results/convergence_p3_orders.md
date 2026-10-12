# P3 convergence — empirical orders

Series at the truth floor (machine-precision solvers, or a solver already at its own floor) show no usable points — that is the finding, not a gap. For the adaptive-GL codes the error falls super-algebraically, so the log-log slope understates the convergence; the figures carry the shape.

- airy_caustic   fastpiclef-mesh       : 0 usable points; best err nan
- airy_caustic   fastpiclef-thimble    : 0 usable points; best err nan
- airy_caustic   plind                 : slope -0.18 over 5 points (err 7.4e-10..9.0e-10, time 4.30e-01..6.63e-01 s)
- airy_y1        fastpiclef-mesh       : 0 usable points; best err nan
- airy_y1        fastpiclef-thimble    : 0 usable points; best err nan
- airy_y1        plind                 : slope -25.10 over 5 points (err 1.6e-10..2.1e-07, time 4.61e-01..5.74e-01 s)
- field3         FIONA                 : slope -1.55 over 12 points (err 4.1e-04..1.8e+01, time 4.58e-03..5.08e+00 s)
- field3         fastpiclef-geometric  : 1 usable points; best err 1.4e-01
- field3         fastpiclef-rc         : slope -4.57 over 3 points (err 1.5e-10..1.8e-07, time 1.32e-01..6.56e-01 s)
- field3         fastpiclef-rc-batch   : 2 usable points; best err 1.5e-10
- nfw            fastpiclef-geometric  : 1 usable points; best err 9.1e-02
- nfw            fastpiclef-rc         : 0 usable points; best err nan
- nfw            fastpiclef-rc-batch   : 0 usable points; best err nan
- point_mass     FIONA                 : slope -1.39 over 12 points (err 2.1e-03..5.4e+01, time 6.42e-03..4.23e+00 s)
- point_mass     fastpiclef-geometric  : 1 usable points; best err 3.3e-04
- point_mass     fastpiclef-rc         : slope -6.39 over 3 points (err 1.1e-11..4.7e-07, time 4.30e-02..2.23e-01 s)
- point_mass     fastpiclef-rc-batch   : 2 usable points; best err 1.5e-11
- rational_k2    P-LPI                 : slope -0.56 over 10 points (err 5.6e-09..5.4e-05, time 9.38e-02..9.61e-01 s)
- rational_k2    fastpiclef-mesh       : 1 usable points; best err 2.0e-11
- rational_k2    fastpiclef-thimble    : 0 usable points; best err nan
- rational_k2    plind                 : slope -5.11 over 4 points (err 3.0e-05..7.9e-02, time 1.96e+00..6.05e+00 s)
- type2_macro    fastpiclef-geometric  : 1 usable points; best err 6.6e-02
- type2_macro    fastpiclef-rc         : slope -12.18 over 3 points (err 9.3e-12..6.0e-07, time 1.31e-01..3.06e-01 s)
- type2_macro    fastpiclef-rc-batch   : 2 usable points; best err 7.8e-12

## Rate vs evaluations (adaptive codes: decades of error per evaluation)
- field3         FIONA               : -8.662e-07 decades per evaluation (12 points, evals 4096..4000000)
- field3         fastpiclef-rc       : -2.557e-03 decades per evaluation (4 points, evals 606..3726)
- field3         fastpiclef-rc-batch : -1.832e-03 decades per evaluation (4 points, evals 620..5260)
- nfw            fastpiclef-rc       : -1.333e-02 decades per evaluation (4 points, evals 366..526)
- nfw            fastpiclef-rc-batch : -1.414e-02 decades per evaluation (4 points, evals 380..540)
- point_mass     FIONA               : -7.268e-07 decades per evaluation (12 points, evals 4096..4000000)
- point_mass     fastpiclef-rc       : -4.172e-03 decades per evaluation (5 points, evals 686..2766)
- point_mass     fastpiclef-rc-batch : -4.255e-03 decades per evaluation (4 points, evals 700..2780)
- rational_k2    P-LPI               : +5.441e-06 decades per evaluation (10 points, evals 512..131072)
- type2_macro    fastpiclef-rc       : -5.477e-03 decades per evaluation (4 points, evals 686..2126)
- type2_macro    fastpiclef-rc-batch : -4.307e-03 decades per evaluation (4 points, evals 700..2780)
