# =====================================================================
# np_helper.py - A scitoolkit file
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
from scitoolkit.util.py23 import *  # py2/3 compatibility

import numpy as np

# Contains NumPy helper functions


def sizeof_ndarray(ndarray):  # TODO unit tests for everything...
    dtype = get_dtype(ndarray)
    if is_int(ndarray, ignore_unsigned=False):
        element_bits = np.iinfo(dtype).bits
    elif is_float(ndarray) or is_complex(ndarray):
        element_bits = np.finfo(dtype).bits
    else:
        raise NotImplementedError('`sizeof` not implemented for `ndarray` of '
                                  'type {}.'.format(dtype))
    element_bytes = element_bits // 8
    n_elements = ndarray.size
    bytes = n_elements * element_bytes
    return bytes


def min_dtype(simple=None, vmin=None, vmax=None, signed=None, floating=None,
              complex_=None):
    if vmin is None and vmax is None:
        if simple is None:
            raise ValueError('`vmin` and/or `vmax` must be specified.')
        dtype = get_dtype(simple)
        if signed is not None:
            if not is_int(simple):
                raise ValueError('`sign` cannot be specified if dtype of '
                                 'simple is {}, which is non-int.'.format(dtype))
            if signed:
                dtype = np.int
            else:
                dtype = np.uint
    else:
        if simple is not None:
            raise ValueError('`simple` cannot be specified with other arguments.')
        if signed is not None:
            if floating is not None:
                raise ValueError('`floating` and `signed` cannot be specified '
                                 'together.')
            if complex_ is not None:
                raise ValueError('`complex_` and `signed` cannot be specified '
                                 'together.')
            if signed:
                dtype = min_int_dtype(vmin=vmin, vmax=vmax)
            else:
                dtype = min_uint_dtype(vmin=vmin, vmax=vmax)
        elif floating is not None:
            if complex_ is not None:
                raise ValueError('`complex_` and `floating` cannot be specified '
                                 'together.')
            dtype = min_float_dtype(vmin=vmin, vmax=vmax)
        elif complex_ is not None:
            dtype = min_complex_dtype(vmin=vmin, vmax=vmax)
        else:
            # Infer dtype from bounds
            if is_complex(vmin) or is_complex(vmax):
                dtype = min_complex_dtype(vmin=vmin, vmax=vmax)
            elif is_float(vmin) or is_float(vmax):
                dtype = min_float_dtype(vmin=vmin, vmax=vmax)
            else:
                dtype = min_int_dtype(vmin=vmin, vmax=vmax, ignore_unsigned=False)
    return dtype


def get_type(value):
    if hasattr(value, 'dtype'):
        return value.dtype
    else:
        return type(value)


def get_dtype(value):
    if hasattr(value, 'dtype'):
        return value.dtype
    else:
        return np.dtype(type(value))


def is_int(value, ignore_unsigned=True):
    dtype = get_dtype(value)
    if ignore_unsigned:
        return np.issubdtype(dtype, np.integer)
    else:
        return (np.issubdtype(dtype, np.integer) or
                np.issubdtype(dtype, np.unsignedinteger))


def is_uint(value):
    return np.issubdtype(get_dtype(value), np.unsignedinteger)


def is_float(value):
    return np.issubdtype(get_dtype(value), np.floating)


def is_complex(value):
    return np.issubdtype(get_dtype(value), np.complex)


def _min_within_dtypes(dtypes, dinfo_func, vmin=None, vmax=None):
    if vmin is None and vmax is None:
        raise ValueError('Either `vmin` or `vmax` must be specified.')
    min_bits = np.inf
    min_dtype_ = None
    for dtype in dtypes:
        dinfo = dinfo_func(dtype)
        if vmin is not None:
            if dinfo.min > vmin:
                continue
        if vmax is not None:
            if dinfo.max < vmax:
                continue
        if dinfo.bits < min_bits:
            min_bits = dinfo.bits
            min_dtype_ = dtype
    if min_dtype_ is None:
        raise ValueError('Specified values vmin={} vmax={} too small/large for '
                         'system dtypes:\n{}'.format(vmin, vmax, dtypes))
    return min_dtype_


def min_float_dtype(vmin=None, vmax=None):
    return _min_within_dtypes(dtypes=np.sctypes['float'],
                              dinfo_func=np.finfo,
                              vmin=vmin, vmax=vmax)


def min_complex_dtype(vmin=None, vmax=None):
    return _min_within_dtypes(dtypes=np.sctypes['complex'],
                              dinfo_func=np.finfo,
                              vmin=vmin, vmax=vmax)


def min_int_dtype(vmin=None, vmax=None, ignore_unsigned=True):
    if ignore_unsigned:
        return _min_within_dtypes(dtypes=np.sctypes['int'],
                                  dinfo_func=np.iinfo,
                                  vmin=vmin, vmax=vmax)
    else:
        try:
            min_dtype_ = _min_within_dtypes(dtypes=np.sctypes['int'],
                                            dinfo_func=np.iinfo,
                                            vmin=vmin, vmax=vmax)
        except ValueError:
            min_dtype_ = _min_within_dtypes(dtypes=np.sctypes['uint'],
                                            dinfo_func=np.iinfo,
                                            vmin=vmin, vmax=vmax)
        return min_dtype_


def min_uint_dtype(vmin=None, vmax=None):
    return _min_within_dtypes(dtypes=np.sctypes['uint'],
                              dinfo_func=np.iinfo,
                              vmin=vmin, vmax=vmax)
