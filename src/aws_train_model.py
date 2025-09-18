#!/usr/bin/env python3
"""
SageMaker training entrypoint: pass-through to train_model.py with supported flags only.
"""

import shlex
import subprocess
import sys
from pathlib import Path

def main():
    script = Path(__file__).parent / "train_model.py"
    cmd = [sys.executable, str(script), *sys.argv[1:]]
    print("Launching:", " ".join(shlex.quote(x) for x in cmd))
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()