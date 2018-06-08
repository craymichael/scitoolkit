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
from scitoolkit.util.py23 import *

import os
import re
from tempfile import mkdtemp, mkstemp
from shutil import rmtree
from dateutil import parser as date_parser
from scitoolkit.util.py_helper import is_str

PATH_REGEX = re.compile(r'[\\/:\"*?<>|]+')
__all__ = ['get_tmp_dir', 'mkdir_tmp', 'TmpDir', 'open_tmp', 'TmpFile',
           'get_tmp_file', 'join', 'get_most_recent_in_dir',
           'get_most_recent_k_in_dir', 'EmptyDirError', 'valid_filename']


class EmptyDirError(Exception):
    pass


def join(*args):
    filename = os.path.join(*args)
    dirname = os.path.dirname(filename)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    return filename


def valid_filename(filename):
    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename)
    # Remove invalid chars (Windows regex)
    basename = PATH_REGEX.sub('_', basename)
    return os.path.join(dirname, basename)


def get_most_recent_in_dir(dirname, delim='_', ext=None, raise_=False,
                           return_fns=False):
    ret = get_most_recent_k_in_dir(dirname, k=1, delim=delim, ext=ext,
                                   raise_=raise_, return_fns=return_fns)
    if return_fns:
        if len(ret[0]):
            return ret[0][0], ret[1]
        else:
            return None, ret[1]
    else:
        if len(ret):
            return ret[0]
        else:
            return None


def get_most_recent_k_in_dir(dirname, k, delim='_', ext=None, raise_=False,
                             strict=False, return_fns=False):
    if is_str(dirname):
        # Note this checks files only.
        if not raise_ and not os.path.isdir(dirname):
            return None

        dirs = os.listdir(dirname)  # Exception can be raised here

        if not dirs:

            if raise_:
                raise EmptyDirError('Directory "{}" cannot be '
                                    'empty.'.format(dirname))
            else:
                return None

        filenames = os.listdir(dirname)
    else:
        # Assume iterable
        filenames = dirname

    fn_dt_dict = {}

    for filename in filenames:

        if os.path.isfile(filename):

            filename_raw = filename

            if ext is not None:
                suffix = '.' + ext
                if not filename.endswith(suffix):
                    continue

                filename = filename[:-len(suffix)]

            if delim is not None:
                split = filename.rsplit(delim, 1)

                if len(split) != 2:
                    continue

                filename = split[1]

            try:
                dt = date_parser.parse(filename)
            except ValueError:
                continue

            fn_dt_dict[filename_raw] = dt

    if not fn_dt_dict and raise_:
        raise EmptyDirError(
            'Directory "{}" does not contain any files with {}{}a valid '
            'timestamp.'.format(dirname,
                                'extension "{}" and '.format(ext) if ext else '',
                                'delimiter "{}" and '.format(delim) if delim else ''))

    most_recent_fns = sorted(fn_dt_dict, key=fn_dt_dict.get, reverse=True)[:k]

    if strict and len(most_recent_fns) != k:
        raise ValueError(
            'Directory "{}" does not contain {} files with {}{}a valid '
            'timestamp.'.format(dirname, k,
                                'extension "{}" and '.format(ext) if ext else '',
                                'delimiter "{}" and '.format(delim) if delim else ''))

    if return_fns:
        return most_recent_fns, list(fn_dt_dict.keys())
    else:
        return most_recent_fns


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
                chosen from a infrastructure-dependent list, but the user of the
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
                chosen from a infrastructure-dependent list, but the user of the
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
