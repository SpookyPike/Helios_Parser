# Live plasmon UI validation

This pass renders the plasmon tab on real datasets and setting combinations to verify that the widget follows the recomputed result rather than stale control state.

| dataset | snapshot | model | exec | runtime [s] | time plot | profile plot | peak line | backend | screenshot |
|---|---:|---|---|---:|---|---|---|---|---|
| 50Al+10E+25CH+3.5TW_stabilized.h5 | 630 | rpa_static_lfc | benchmark | 2.50 | LOS-integrated plasmon spectrum | Local k*lambda_D profile / snapshot 630 @ 6.300 ns | Peak dE       21.47 eV | Backend       classical_maxwellian | `outputs\validation_outputs\plasmon_ui_live\50Al_driven_rpa_static_lfc.png` |
| 50Al+10E+25CH+3.5TW_stabilized.h5 | 630 | lindhard | benchmark | 3.62 | LOS-integrated plasmon spectrum | Local k*lambda_D profile / snapshot 630 @ 6.300 ns | Peak dE       24.61 eV | Backend       finite_t_lindhard | `outputs\validation_outputs\plasmon_ui_live\50Al_driven_lindhard.png` |
| Cu_0166_stabilized.h5 | 0 | quicklook | quicklook | 0.14 | Plasma frequency vs time | Local k*lambda_D profile / snapshot 0 @ 1.000 fs | Peak dE       - eV | Backend       classical_maxwellian | `outputs\validation_outputs\plasmon_ui_live\Cu0166_quicklook.png` |
| 5Fe+4.9TW+light_stabilized.h5 | 5 | mermin_static_lfc | benchmark | 0.02 | Plasma frequency vs time | Local k*lambda_D profile / snapshot 5 @ 0.500 ns | Peak dE       - eV | Backend       classical_maxwellian | `outputs\validation_outputs\plasmon_ui_live\Fe_light_mermin_static_lfc.png` |

Checks performed:
- switched between quicklook, RPA-static-LFC, Lindhard, and Mermin-static-LFC on real HDF5 data
- verified the summary line updates with the applied model/backend
- verified that benchmark spectral models switch the profile panel to the observed spectrum, while quicklook/non-spectral states stay on state profiles
- captured screenshots after each refresh for manual inspection