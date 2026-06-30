"""
verify.py — legacy entry point; delegates to waam_twin v2 validation suite.
"""

import sys

from waam_twin.validation.run_all import main

if __name__ == "__main__":
    sys.exit(main())
