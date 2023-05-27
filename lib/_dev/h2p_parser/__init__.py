"""
h2p_parser

Heteronym to Phoneme Parser

"""

import sys

if sys.version_info < (3, 9):
    # In Python versions below 3.9, this is needed
    from importlib_resources import files
else:
    # Since python 3.9+, importlib.resources.files is built-in
    from importlib.resources import files

__version__ = "1.0.0"

# Data module
DATA_PATH = files(__name__ + '.data')
# Iterable collection of all files in data.
DATA_FILES = DATA_PATH.iterdir()
