
import gc
import sys
import unittest
from PySide6 import QtWidgets
from _viewer_test_utils import get_app, process_events
import test_viewer_payloads as m

app = get_app()
all_names = [
    'test_dynamic_radius_mode_stays_disabled_until_radius_payload_is_loaded',
    'test_2d_orientation_and_coordinate_payloads_match_reader',
    'test_snapshot_lineouts_and_time_traces_match_reader',
    'test_region_and_material_masks_apply_only_along_coordinate_dimension',
    'test_boundary_overlays_and_scale_modes_match_expected_payloads',
    'test_new_helios_format_field_and_diagnostic_loading_remain_consistent',
]
for name in all_names:
    print('START', name, flush=True)
    suite = unittest.TestSuite([m.ViewerPayloadTests(name)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    process_events(100)
    widgets = [w for w in app.topLevelWidgets() if not w.objectName().startswith('qt_')]
    print('RESULT', name, result.wasSuccessful(), 'widgets', len(widgets), flush=True)
    for w in widgets:
        try:
            print('  widget', type(w).__name__, 'visible', w.isVisible(), 'title', getattr(w, 'windowTitle', lambda: '')(), flush=True)
        except Exception:
            pass
    gc.collect()
    process_events(100)
    if not result.wasSuccessful():
        sys.exit(1)
print('DONE', flush=True)
