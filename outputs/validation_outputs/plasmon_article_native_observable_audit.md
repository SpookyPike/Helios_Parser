# Article-native Al XRTS observable audit

The minimal XRTS observable seam already established that the remaining driven residual is not removed by simply wrapping the backend DSF with a free+elastic proxy.

Before the article-native layer, the observable path still had four concrete limitations:
- elastic/ion feature was represented only by a compact generic Al proxy, not an explicit Al atomic-form-factor split
- bound/core inelastic contribution existed only as placeholder bookkeeping
- density-averaged representative spectra dropped the component arrays during export, which made component-breakdown CSVs misleading
- density-averaged observable peak extraction could fail catastrophically at high q because the local quadratic fit was accepted even when it detached from the true inelastic maximum

This pass upgrades the observable construction specifically for Al:
- free-electron inelastic term still comes from the validated backend DSF (QHD, STLS, RPA, RPA+static LFC)
- elastic feature now uses explicit Al Cromer-Mann form factors
- neutral-Al and Al3+ form-factor bookkeeping is separated so the screening/core split is visible in provenance
- bound/core inelastic contribution is shell-resolved and explicitly zero below the Al L-shell onset in the current 45 eV benchmark window
- article-facing comparison is made on the positive inelastic branch after explicit elastic subtraction rather than on the raw total spectrum peak

Honesty limits remain explicit:
- ion structure factor is still treated with the unity assumption
- no full bound-free atomic cross section is introduced
- no hidden rescaling or normalization fit is applied
- exact article-side background subtraction and detector processing assumptions are still not recoverable from the current repo assets
