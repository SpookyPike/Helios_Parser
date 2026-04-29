# HELIOS Parse / View 1.0.1

Code developed by Dmitrii Bespalov at European XFEL.
Release date: 2026-04-29.

## Open this first

This bundle contains the current HELIOS Parse / View source tree plus documentation, launch scripts, and a few demo HDF5 runs.

Production GUI and backend compute expose only physically gated v1 workflows by default. Experimental Plasmon/XRTS and Transmission workflows remain available only when `HELIOS_DEV_MODE=1` or `HELIOS_ENABLE_EXPERIMENTAL=1` is set.
The bundle includes the XCOM support artifacts expected by the development/experimental code paths:

- `x-com_fallback/` with the precomputed cold-XCOM attenuation tables
- `helios_xcom_integration.zip` with the Python wrapper/backend integration package
- `XCOM.tar.gz` with the vendor XCOM source archive

## Windows

1. Open PowerShell in this folder.
2. Run:

```powershell
.\Run_HELIOS_Analyzer.ps1
```

The script will:

- create a local `.venv` if needed
- install the desktop dependencies
- launch the application

## Linux

1. Open a terminal in this folder.
2. Run:

```bash
chmod +x ./run_helios_analyzer.sh
./run_helios_analyzer.sh
```

The script will:

- create a local `.venv` if needed
- install the desktop dependencies
- launch the application

## Included examples

Look in the `examples/` folder for small and moderate demo HDF5 runs that are appropriate for onboarding.

## Documentation

Start with:

- `README.md`
- `docs/index.html`
- `docs/user-guide.html`

## Notes

- Python 3.10 or newer is required.
- The bundle is source-based, not a frozen binary package.
- Larger layered advanced-analysis datasets are documented with screenshots, but not all of them are bundled to keep the archive practical.
