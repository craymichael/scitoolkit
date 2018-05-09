from scitoolkit.py23 import *  # py2/3 compatibility

import os


def sys_has_display():
    return 'DISPLAY' in os.environ.keys()
