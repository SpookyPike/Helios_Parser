from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np

import _test_bootstrap  # noqa: F401
from helios_parser import HeliosParser, HeliosPreview, inspect, parse, preview, write_hdf5


ROOT = Path(__file__).resolve().parents[1]
TRACKED_SAMPLE = ROOT / "new_data" / "25Cu+1.87TW" / "25Cu+1.87TW.log"


def main() -> None:
    parser = HeliosParser()
    mmap_parser = HeliosParser(access_mode="mmap")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        log_only = tmpdir_path / TRACKED_SAMPLE.name
        log_only.write_bytes(TRACKED_SAMPLE.read_bytes())
        header = inspect(log_only)
        preview_result = preview(log_only)
        mmap_preview = mmap_parser.preview(log_only)
        assert header.n_zones == 50
        assert isinstance(preview_result, HeliosPreview)
        assert preview_result.snapshot is not None
        assert mmap_preview.snapshot is not None
        assert preview_result.snapshot.fields["density"].shape == (50,)
        np.testing.assert_allclose(preview_result.snapshot.fields["density"], mmap_preview.snapshot.fields["density"])

        simulation = parse(log_only)
        assert simulation.metadata["n_zones"] == 50
        assert simulation.time["time"].size > 1
        assert simulation.fields["density"].shape == (simulation.time["time"].size, 50)
        assert simulation.fields["temperature_e"].shape == simulation.fields["density"].shape
        assert np.all(np.isfinite(simulation.grid["zone_mass"]))
        assert "pressure" in simulation.fields
        assert set(simulation.diagnostics) == {
            "radiation_boundary_fluxes",
            "energy_summary",
            "energy_exchange",
            "energy_balance",
        }
        assert set(simulation.input_parameters) == {
            "hydro",
            "laser_source",
            "radiation_source",
            "radiative_transfer",
            "time_control",
        }

        output = Path(tmpdir) / "helios.h5"
        write_hdf5(log_only, output, overwrite=True, parser=parser)
        with h5py.File(output, "r") as handle:
            assert "/grid/x" in handle
            assert "/time/time" in handle
            assert "/fields/density" in handle
            assert "/regions/region_index" in handle
            assert "/materials/index" in handle
            assert "/diagnostics/energy_summary/current/ions" in handle
            assert "/metadata/input_parameters/hydro/plasma_model" in handle
            assert handle["/grid/x"].attrs["unit"] == "cm"
            assert handle["/grid/x"].attrs["units"] == "cm"
            assert handle["/fields/density"].attrs["unit"] == "g/cm3"
            assert handle["/fields/density"].attrs["units"] == "g/cm3"
            assert handle["/diagnostics/energy_summary/current/ions"].attrs["units"] == "J/cm**2"
            assert handle["/fields/density"].shape[1] == 50

    print("smoke test passed")


if __name__ == "__main__":
    main()
