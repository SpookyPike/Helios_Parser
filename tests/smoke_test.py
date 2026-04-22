from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np

import _test_bootstrap  # noqa: F401
from helios_parser import HeliosParser, HeliosPreview, inspect, parse, preview, write_hdf5


ROOT = Path(__file__).resolve().parents[1]
SMALLEST = ROOT / "5Fe+4.9TW+light.log"
VALIDATION = ROOT / "25Cu+1.4TW.log"


def main() -> None:
    parser = HeliosParser()
    mmap_parser = HeliosParser(access_mode="mmap")
    header = inspect(SMALLEST)
    preview_result = preview(SMALLEST)
    mmap_preview = mmap_parser.preview(SMALLEST)
    assert header.n_zones == 500
    assert isinstance(preview_result, HeliosPreview)
    assert preview_result.snapshot is not None
    assert mmap_preview.snapshot is not None
    assert preview_result.snapshot.fields["density"].shape == (500,)
    np.testing.assert_allclose(preview_result.snapshot.fields["density"], mmap_preview.snapshot.fields["density"])

    smallest = parse(SMALLEST)
    assert smallest.metadata["n_zones"] == 500
    assert smallest.time["time"].size > 1
    assert smallest.fields["density"].shape == (smallest.time["time"].size, 500)
    assert smallest.fields["temperature_e"].shape == smallest.fields["density"].shape
    assert np.all(np.isfinite(smallest.grid["zone_mass"]))
    assert "pressure" in smallest.fields
    assert set(smallest.diagnostics) == {
        "radiation_boundary_fluxes",
        "energy_summary",
        "energy_exchange",
        "energy_balance",
    }
    assert set(smallest.input_parameters) == {
        "hydro",
        "laser_source",
        "radiation_source",
        "radiative_transfer",
        "time_control",
    }

    validation = parser.parse(VALIDATION)
    assert validation.metadata["n_zones"] == 50
    assert validation.time["time"].size > 1
    assert validation.fields["density"].shape == (validation.time["time"].size, 50)
    assert smallest.raw_field_map.keys() == validation.raw_field_map.keys()

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "helios.h5"
        write_hdf5(SMALLEST, output, overwrite=True, parser=parser)
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
            assert handle["/fields/density"].shape[1] == 500

    print("smoke test passed")


if __name__ == "__main__":
    main()
