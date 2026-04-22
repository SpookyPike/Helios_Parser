# Plasmon article-level observable audit

The current dielectric benchmark path compares a peak extracted from the backend free-electron loss/DSF response.

The new observable path raises the comparison level to a minimal experiment-facing XRTS reconstruction:
- free-electron inelastic term from the selected backend (RPA, static-LFC, QHD, STLS)
- explicit central elastic/ion-feature proxy
- explicit bound/core term bookkeeping
- instrument convolution before peak extraction

Implemented decomposition:
- minimal Chihara-like Al observable
- Al free term comes directly from the backend DSF
- elastic term uses a bound-electron form-factor proxy centered at zero energy transfer
- bound/core inelastic term is kept explicit but currently zeroed in the narrow article benchmark window below the first Al L-shell onset

Important honesty constraints:
- this is not article-native atomic physics
- no hidden normalization fit is applied
- mixed/unsupported materials fall back to the backend free-electron spectrum with explicit provenance
- the observable peak is extracted from the positive branch after excluding the elastic core window

Expected leverage:
- if residual is mostly observable-level, XRTS mode should improve experiment-facing MAE without materially changing backend-matched dielectric-branch trends
- if residual stays large, the missing physics is likely bound-electron / atomic-form-factor / article-native comparison structure rather than another dielectric tweak
