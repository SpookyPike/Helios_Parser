
import faulthandler
import sys
import unittest
from pathlib import Path

log_path = Path('outputs') / 'trace_viewer_payloads_run.log'
log_path.parent.mkdir(parents=True, exist_ok=True)
log = log_path.open('w', encoding='utf-8', buffering=1)
faulthandler.enable(log)

class LoggingResult(unittest.TextTestResult):
    def startTest(self, test):
        log.write(f'START {test.id()}\n')
        log.flush()
        super().startTest(test)
    def addSuccess(self, test):
        log.write(f'OK {test.id()}\n')
        log.flush()
        super().addSuccess(test)
    def addFailure(self, test, err):
        log.write(f'FAIL {test.id()}\n')
        log.flush()
        super().addFailure(test, err)
    def addError(self, test, err):
        log.write(f'ERROR {test.id()}\n')
        log.flush()
        super().addError(test, err)

class LoggingRunner(unittest.TextTestRunner):
    resultclass = LoggingResult

suite = unittest.defaultTestLoader.loadTestsFromName('test_viewer_payloads')
result = LoggingRunner(verbosity=2).run(suite)
log.write(f'DONE success={result.wasSuccessful()}\n')
log.flush()
log.close()
sys.exit(0 if result.wasSuccessful() else 1)
