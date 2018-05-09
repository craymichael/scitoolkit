# =====================================================================
# file_system.py - A scitoolkit file
# Copyright (C) 2018  Zach Carmichael
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# =====================================================================
from scitoolkit.py23 import *

import os
from tempfile import mkdtemp, mkstemp
from shutil import rmtree

__all__ = ['get_tmp_dir', 'mkdir_tmp', 'TmpDir', 'open_tmp', 'TmpFile',
           'get_tmp_file']


def get_tmp_dir(suffix='', prefix='tmp', dir=None):
    """
    (5/9/18) Arg descriptions taken from:
        https://docs.python.org/2/library/tempfile.html

    Args:
        suffix: If suffix is specified, the file name will end with that suffix,
                otherwise there will be no suffix. mkstemp() does not put a dot
                between the file name and the suffix; if you need one, put it at
                the beginning of suffix.
        prefix: If prefix is specified, the file name will begin with that prefix;
                otherwise, a 'tmp' is used.
        dir:    If dir is specified, the file will be created in that directory;
                otherwise, a default directory is used. The default directory is
                chosen from a platform-dependent list, but the user of the
                application can control the directory location by setting the
                TMPDIR, TEMP or TMP environment variables. There is thus no
                guarantee that the generated filename will have any nice properties,
                such as not requiring quoting when passed to external commands via
                os.popen().
     Returns:
         Absolute pathname of the new directory
    """
    return mkdtemp(suffix=suffix, prefix=prefix, dir=dir)


class TmpDir(object):
    def __init__(self, *args, ignore_errors=False, **kwargs):
        self.dir_name = get_tmp_dir(*args, **kwargs)
        # Set to True to not fail on deleting read-only files
        self._ignore_errors = ignore_errors

    def __enter__(self):
        return self.dir_name

    def __exit__(self, exc_type, exc_val, exc_tb):
        rmtree(self.dir_name, ignore_errors=self._ignore_errors)


def mkdir_tmp(*args, **kwargs):
    return TmpDir(*args, **kwargs)


def get_tmp_file(suffix='', prefix='tmp', dir=None, text=False):
    """
    (5/9/18) Arg descriptions taken from:
        https://docs.python.org/2/library/tempfile.html

    Args:
        suffix: If suffix is specified, the file name will end with that suffix,
                otherwise there will be no suffix. mkstemp() does not put a dot
                between the file name and the suffix; if you need one, put it at
                the beginning of suffix.
        prefix: If prefix is specified, the file name will begin with that prefix;
                otherwise, a 'tmp' is used.
        dir:    If dir is specified, the file will be created in that directory;
                otherwise, a default directory is used. The default directory is
                chosen from a platform-dependent list, but the user of the
                application can control the directory location by setting the
                TMPDIR, TEMP or TMP environment variables. There is thus no
                guarantee that the generated filename will have any nice properties,
                such as not requiring quoting when passed to external commands via
                os.popen().
        text:   If text is specified, it indicates whether to open the file in
                binary mode (the default) or text mode. On some platforms, this
                makes no difference.

     Returns:
         Tuple of (open OS-level file handle, absolute pathname of the new file)
    """
    return mkstemp(suffix=suffix, prefix=prefix, dir=dir, text=text)


class TmpFile(object):
    def __init__(self, *args, **kwargs):
        self.f, self.file_name = get_tmp_file(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.remove(self.file_name)


def open_tmp(*args, **kwargs):
    return TmpFile(*args, **kwargs)
