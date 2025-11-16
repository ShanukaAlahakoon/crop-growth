import importlib
import sys
import traceback

try:
    importlib.import_module('app')
    print('MODEL LOAD: OK')
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
