"""Entry point for the standalone binaries built by .github/workflows/release.yml."""

import multiprocessing
import sys

from clackless.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
