"""Release version constants, free of numpy, h5py and Qt.

WHY THIS IS TOP-LEVEL.  Importing anything under ``helios_parser`` or
``helios_app`` first executes that package's ``__init__``, which pulls numpy,
h5py and Qt: measured 228 ms for ``helios_parser`` and 986 ms for
``helios_app``.  A module at the package root imports none of them, so the
command line can answer ``--version`` in milliseconds instead of importing the
world first.

``helios_app.release`` re-exports these names and remains the public face, so
the packaging scripts and the About dialog are unchanged.
"""

from __future__ import annotations

DISTRIBUTION_NAME = "helios-parser"
APP_NAME = "HELIOS Parse / View"
RELEASE_VERSION = "2.1.0"
RELEASE_DATE = "2026-11-04"