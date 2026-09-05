import os
import sys

# Allow tests/test_module.py to import inference.py and service.py from the
# parent directory without needing to install this as a package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
