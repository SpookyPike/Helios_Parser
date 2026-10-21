"""Bounded subprocess adapter for the bundled NIST XCOM 3.1 program."""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import numpy as np

from helios.cancellation import AnalysisCancelled
from helios.services.derived.xcom_composition import composition_from_formula


_BATCH_SIZE = 64
_COEFFICIENT_CACHE_LIMIT = 65536
_COEFFICIENT_CACHE_MAX_BYTES = 16 * 1024 * 1024
_TIMEOUT_SECONDS = 10
_NUMBER = re.compile(r"[-+]?\d+\.\d+[EDed][-+]\d+")
_ELEMENT_SYMBOL = re.compile(r"[A-Z][a-z]?")
_CLIENT_LOCK = threading.Lock()
_DEFAULT_CLIENT = None
defer_compute_probe = True
_FIELDS = (
    "energy_mev",
    "coherent_cm2_g",
    "incoherent_cm2_g",
    "photoelectric_cm2_g",
    "pair_nuclear_cm2_g",
    "pair_electron_cm2_g",
    "mu_rho_total_cm2_g",
    "mu_rho_no_coherent_cm2_g",
)


def _platform_folder() -> str:
    system = "windows" if os.name == "nt" else sys.platform
    machine = {"AMD64": "x86_64", "x86_64": "x86_64", "aarch64": "arm64"}.get(
        os.uname().machine if hasattr(os, "uname") else os.environ.get("PROCESSOR_ARCHITECTURE", ""),
        "unknown",
    )
    return f"{system}-{machine}"


def provider_paths() -> tuple[Path | None, Path | None]:
    """Resolve explicit configuration first, then a frozen application bundle."""
    executable = os.environ.get("HELIOS_XCOM_EXECUTABLE")
    data_directory = os.environ.get("HELIOS_XCOM_DATA_DIR")
    if executable or data_directory:
        return (Path(executable).expanduser() if executable else None,
                Path(data_directory).expanduser() if data_directory else None)
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        root = Path(bundle_root) / "helios_xcom_native" / _platform_folder()
        name = "xcom.exe" if os.name == "nt" else "xcom"
        return root / name, root
    return None, None


@lru_cache(maxsize=8)
def _validated_provider_identity(executable: str, data_directory: str) -> str | None:
    executable_path = Path(executable)
    data_path = Path(data_directory)
    try:
        manifest = json.loads((data_path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("engine") != "NIST XCOM 3.1":
            return None
        if manifest.get("energy_range_kev") != [1.0, 35.0]:
            return None
        executable_digest = hashlib.sha256(executable_path.read_bytes()).hexdigest()
        if executable_digest != manifest.get("executable_sha256"):
            return None
        expected_data = manifest.get("element_data_sha256", {})
        if len(expected_data) != 100:
            return None
        digest = hashlib.sha256()
        digest.update(executable_digest.encode("ascii"))
        for index in range(1, 101):
            name = f"MDATX3.{index:03d}"
            data_file = data_path / name
            data_digest = hashlib.sha256(data_file.read_bytes()).hexdigest()
            if expected_data.get(name) != data_digest:
                return None
            digest.update(f"{name}:{data_digest}\n".encode("ascii"))
        return digest.hexdigest()
    except OSError:
        return None


def provider_identity() -> str | None:
    executable, data_directory = provider_paths()
    if executable is None or data_directory is None:
        return None
    return _validated_provider_identity(str(executable.resolve()), str(data_directory.resolve()))


def _parse_rows(output: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for line in output.splitlines():
        matches = list(_NUMBER.finditer(line))
        if len(matches) != len(_FIELDS):
            continue
        values = [float(match.group().replace("D", "E").replace("d", "e")) for match in matches]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("XCOM output contains a non-finite value.")
        if any(value < 0 for value in values[1:]):
            raise ValueError("XCOM output contains a negative coefficient.")
        row: dict[str, float | str] = dict(zip(_FIELDS, values))
        row["edge_label"] = line[:matches[0].start()].strip()
        rows.append(row)
    if not rows:
        raise ValueError("XCOM output contained no coefficient rows.")
    return rows


@dataclass(frozen=True, slots=True)
class XcomProvider:
    executable: Path
    data_directory: Path
    fingerprint: str


def resolve_provider() -> XcomProvider:
    executable, data_directory = provider_paths()
    if executable is None or data_directory is None:
        raise FileNotFoundError(
            "Native XCOM is not installed. Configure HELIOS_XCOM_EXECUTABLE and HELIOS_XCOM_DATA_DIR."
        )
    executable = executable.resolve()
    data_directory = data_directory.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"XCOM executable not found: {executable}")
    missing = [f"MDATX3.{index:03d}" for index in range(1, 101)
               if not (data_directory / f"MDATX3.{index:03d}").is_file()]
    if missing:
        raise FileNotFoundError(f"XCOM data directory is incomplete ({len(missing)} element files missing).")
    fingerprint = provider_identity()
    if fingerprint is None:
        raise OSError("XCOM provider files could not be fingerprinted.")
    return XcomProvider(executable, data_directory, fingerprint)


class NativeXcomClient:
    """Run one legacy XCOM process at a time in a private, disposable directory."""

    def __init__(self, provider: XcomProvider | None = None) -> None:
        self.provider = provider or resolve_provider()
        self.backend = SimpleNamespace(
            backend_name="Native XCOM 3.1",
            backend_fingerprint=self.provider.fingerprint,
            available=True,
        )
        self._run_lock = threading.Lock()
        self._coefficient_lock = threading.Lock()
        self._coefficients: OrderedDict[tuple[str, str], float] = OrderedDict()
        self._coefficient_bytes = 0
        self._temporary_root = tempfile.TemporaryDirectory(prefix="helios-xcom-data-")
        self._work_directory = Path(self._temporary_root.name)
        for index in range(1, 101):
            shutil.copyfile(
                self.provider.data_directory / f"MDATX3.{index:03d}",
                self._work_directory / f"MDATX3.{index:03d}",
            )

    def close(self) -> None:
        self._temporary_root.cleanup()

    def query_element(
        self,
        symbol: str,
        energies_kev: tuple[float, ...],
        *,
        progress_check=None,
    ) -> dict[float, dict[str, float | str]]:
        if not _ELEMENT_SYMBOL.fullmatch(symbol):
            raise ValueError(f"Invalid XCOM element symbol {symbol!r}.")
        energies = tuple(float(value) for value in energies_kev)
        if not energies or len(energies) > _BATCH_SIZE:
            raise ValueError("Native XCOM requests must contain between 1 and 64 energies.")
        if any(not math.isfinite(value) or not 1.0 <= value <= 35.0 for value in energies):
            raise ValueError("Native XCOM energies must be finite and within 1–35 keV.")
        if energies != tuple(sorted(set(energies))):
            raise ValueError("Native XCOM energies must be unique and sorted.")
        prompts = [symbol, "2", symbol, "3", "3", "1", str(len(energies))]
        prompts.extend(f"{energy * 1.0e-3:.16e}" for energy in energies)
        prompts.extend(("N", "out.txt", "1"))
        with self._run_lock:
            output_path = self._work_directory / "out.txt"
            output_path.unlink(missing_ok=True)
            try:
                process = subprocess.Popen(
                    [str(self.provider.executable)],
                    cwd=self._work_directory,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    shell=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    env={**os.environ, "LC_ALL": "C", "LANG": "C"},
                )
                deadline = time.monotonic() + _TIMEOUT_SECONDS
                pending_input: str | None = "\n".join(prompts) + "\n"
                try:
                    while True:
                        if progress_check is not None:
                            progress_check()
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(
                                f"XCOM did not finish within {_TIMEOUT_SECONDS} seconds."
                            )
                        try:
                            stdout, stderr = process.communicate(
                                input=pending_input,
                                timeout=min(0.1, remaining),
                            )
                            break
                        except subprocess.TimeoutExpired:
                            pending_input = None
                except (AnalysisCancelled, TimeoutError):
                    process.terminate()
                    try:
                        process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    process.communicate()
                    raise
            except OSError as exc:
                raise RuntimeError(f"Could not start the XCOM executable: {exc}") from exc
            if process.returncode != 0 or not output_path.is_file():
                diagnostic = (stdout + "\n" + stderr)[-4000:].strip()
                raise RuntimeError(f"XCOM failed (exit {process.returncode}): {diagnostic}")
            rows = _parse_rows(output_path.read_text(encoding="ascii", errors="strict"))
        if len(rows) != len(energies):
            raise ValueError(f"XCOM returned {len(rows)} rows for {len(energies)} requested energies.")
        result: dict[float, dict[str, float | str]] = {}
        for requested, row in zip(energies, rows):
            returned = float(row["energy_mev"]) * 1000.0
            if not math.isclose(returned, requested, rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError("XCOM output energies do not match the request.")
            row["energy_kev"] = requested
            result[requested] = row  # type: ignore[assignment]
        return result

    def query_total(self, formula: str, energies_kev: tuple[float, ...], *, progress_check=None) -> np.ndarray:
        composition = composition_from_formula(formula)
        total = np.zeros(len(energies_kev), dtype=np.float64)
        grouped: dict[str, list[int]] = {}
        for energy_index, energy in enumerate(energies_kev):
            grouped.setdefault(float(energy).hex(), []).append(energy_index)
        unique_energies = tuple(sorted(float.fromhex(key) for key in grouped))
        coefficients = np.zeros(len(unique_energies), dtype=np.float64)
        for symbol, fraction in composition.mass_fractions:
            if progress_check is not None:
                progress_check()
            component = np.empty(len(unique_energies), dtype=np.float64)
            with self._coefficient_lock:
                missing: list[float] = []
                for index, energy in enumerate(unique_energies):
                    key = (symbol, energy.hex())
                    value = self._coefficients.get(key)
                    if value is None:
                        missing.append(energy)
                    else:
                        self._coefficients.move_to_end(key)
                        component[index] = value
                for start in range(0, len(missing), _BATCH_SIZE):
                    batch = tuple(missing[start:start + _BATCH_SIZE])
                    rows = self.query_element(symbol, batch, progress_check=progress_check)
                    for energy in batch:
                        key = (symbol, energy.hex())
                        value = float(rows[energy]["mu_rho_total_cm2_g"])
                        existing_value = self._coefficients.pop(key, None)
                        if existing_value is not None:
                            self._coefficient_bytes -= (
                                sys.getsizeof(key)
                                + sys.getsizeof(key[0])
                                + sys.getsizeof(key[1])
                                + sys.getsizeof(existing_value)
                            )
                        self._coefficients[key] = value
                        self._coefficients.move_to_end(key)
                        self._coefficient_bytes += (
                            sys.getsizeof(key)
                            + sys.getsizeof(key[0])
                            + sys.getsizeof(key[1])
                            + sys.getsizeof(value)
                        )
                        while (
                            len(self._coefficients) > _COEFFICIENT_CACHE_LIMIT
                            or self._coefficient_bytes > _COEFFICIENT_CACHE_MAX_BYTES
                        ):
                            old_key, old_value = self._coefficients.popitem(last=False)
                            self._coefficient_bytes -= (
                                sys.getsizeof(old_key)
                                + sys.getsizeof(old_key[0])
                                + sys.getsizeof(old_key[1])
                                + sys.getsizeof(old_value)
                            )
                for index, energy in enumerate(unique_energies):
                    component[index] = self._coefficients[(symbol, energy.hex())]
            coefficients += fraction * component
        for unique_index, energy in enumerate(unique_energies):
            for result_index in grouped[energy.hex()]:
                total[result_index] = coefficients[unique_index]
        return total


def default_client() -> NativeXcomClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is not None:
        return _DEFAULT_CLIENT
    with _CLIENT_LOCK:
        if _DEFAULT_CLIENT is None:
            _DEFAULT_CLIENT = NativeXcomClient()
    return _DEFAULT_CLIENT


def compute_multizone_transmission(
    client: NativeXcomClient,
    *,
    material_specs: list[str],
    density_g_cm3: list[float],
    path_length_cm: list[float],
    energies_kev: tuple[float, ...] | list[float],
    progress_check=None,
) -> SimpleNamespace:
    if not (len(material_specs) == len(density_g_cm3) == len(path_length_cm)):
        raise ValueError("Material, density, and path vectors must have equal lengths.")
    energies = tuple(float(value) for value in energies_kev)
    if not energies or any(not math.isfinite(value) or not 1.0 <= value <= 35.0 for value in energies):
        raise ValueError("Native XCOM energies must be finite and within 1–35 keV.")
    grouped: dict[str, float] = {}
    for material, density, length in zip(material_specs, density_g_cm3, path_length_cm):
        rho, path = float(density), float(length)
        if not math.isfinite(rho) or rho < 0 or not math.isfinite(path) or path < 0:
            raise ValueError("Density and path length must be finite and nonnegative.")
        label = str(material).strip()
        grouped[label] = grouped.get(label, 0.0) + rho * path
    optical_depth = np.zeros(len(energies), dtype=np.float64)
    budgets = []
    for formula, areal_density in sorted(grouped.items()):
        if progress_check is not None:
            progress_check()
        coefficient = client.query_total(formula, energies, progress_check=progress_check)
        component_tau = coefficient * areal_density
        optical_depth += component_tau
        budgets.append(SimpleNamespace(
            label=formula,
            areal_density_g_cm2=areal_density,
            optical_depth=component_tau,
        ))
    return SimpleNamespace(
        energies_kev=energies,
        transmission=np.exp(-optical_depth),
        optical_depth=optical_depth,
        per_material=budgets,
        attenuation_mode="total_with_coherent",
    )
