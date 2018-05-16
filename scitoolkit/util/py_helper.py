# =====================================================================
# py_helper.py - A scitoolkit file
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

from six import string_types, iteritems, itervalues  # py2/3 compatibility
import sys  # version info
from copy import deepcopy
from inspect import getfullargspec
from types import ModuleType


def is_py3():
    return sys.version_info.major == 3


def is_py2():
    return sys.version_info.major == 2


def is_str(s):
    return isinstance(s, string_types)


def hashable(o):
    try:
        hash(o)
        return True
    except Exception:
        return False


def iterable(o):
    try:
        iter(o)
        return True
    except Exception:
        return False


def can_reverse_dict(d):
    for v in itervalues(d):
        if not hashable(v):
            return False
    return True


def reverse_dict(d, copy=False):
    if copy:
        d = deepcopy(d)
    return {v: k for k, v in iteritems(d)}


def func_arg_names(func):
    argspec = getfullargspec(func)
    # Get arg names and kw-only arg names (e.g. if *args specified in function)
    all_arg_names = argspec.args + argspec.kwonlyargs
    return all_arg_names


def get_default_args(func):
    args, varargs, keywords, defaults, _, _, _ = getfullargspec(func)
    return dict(zip(args[-len(defaults):], defaults))


def filter_unused_kwargs(func, kwargs):
    all_arg_names = func_arg_names(func)
    filtered = {}
    for k, v in kwargs:
        if k in all_arg_names:
            filtered[k] = v
    return filtered


def merge_list_of_dicts(dl, hard=True, recurse=False, copy=True):
    if copy:
        dl = deepcopy(dl)
    d0 = None
    l0 = []
    for d in dl:
        if not isinstance(d, dict):
            if hard:
                raise ValueError('`hard` specified but list item {} is of type '
                                 '{}.'.format(d, type(d)))
            else:
                l0.append(d)
                continue
        if recurse:
            dr = {}
            for k, v in iteritems(d):
                if isinstance(v, (tuple, list)):
                    dr[k] = merge_list_of_dicts(v, hard, recurse, copy)
                else:
                    dr[k] = v
            d = dr
        if d0 is None:
            d0 = d
        else:
            d0.update(d)
    if l0:
        if d0 is not None:
            l0.append(d0)
        return l0
    else:
        return d0


def _get_attr_dict_list(module, count, req_all_attr, prefer_all_attr,
                        exclude_childless_modules, skip_levels, filter_func,
                        _level=None):
    if _level is None:
        _level = 0

    attr_dict_list = []
    if count is not None:
        if count == 0:
            return attr_dict_list
        else:
            count -= 1

    if prefer_all_attr and _level not in skip_levels:
        if hasattr(module, '__all__'):
            all_attrs = list(deepcopy(module.__all__))
        else:
            all_attrs = dir(module)
    elif req_all_attr and _level not in skip_levels:
        if hasattr(module, '__all__'):
            all_attrs = list(deepcopy(module.__all__))
        else:
            return attr_dict_list
    else:
        all_attrs = dir(module)

    for attr_name in all_attrs:
        if hasattr(module, attr_name):
            attr = getattr(module, attr_name)
        else:
            attr = None  # TODO...
        if isinstance(attr, ModuleType):
            dl = _get_attr_dict_list(attr, count, req_all_attr, prefer_all_attr,
                                     exclude_childless_modules, skip_levels,
                                     filter_func, _level=_level + 1)
            if not exclude_childless_modules or dl:
                attr_dict_list.append({attr_name: dl})
        elif skip_levels is None or _level not in skip_levels:
            if filter_func is None or filter_func(attr_name, attr):
                attr_dict_list.append(attr_name)

    return attr_dict_list


def get_module_api(module, depth=None, req_all_attr=False, prefer_all_attr=False,
                   exclude_childless_modules=False, skip_levels=None, filter_func=None,
                   merge_dicts='soft'):
    """ Searches recursively through a module instance to uncover its "API", i.e.
    available attributes. Parameters can be used to meaningfully extract useful
    submodules and functions.

    Args:
        module:          The module instance to search
        depth:           The number of modules deep to recurse through
        req_all_attr:    Require that all modules to be considered have the __all__
                         attribute specified.
        prefer_all_attr: Prefer that modules have __all__ specified, and use dir(module)
                         if module does not have __all__ attribute.
        exclude_childless_modules: \
                         Exclude modules that don't have any child attributes.
        skip_levels:     Levels of depth to be excluded as a list (skip_levels=[0]
                         skips the first level).
        filter_func:     Function to filter non-module attributes. Takes two arguments:
                         `attr_name` and `attr`, and returns boolean (False->filter)
        merge_dicts:     How to merge dictionaries together {'soft', 'hard', None,
                         False}

    Returns:
        list of dict objects mapping modules to attributes
    """
    if not isinstance(module, ModuleType):
        raise TypeError('`module` must be a module instance. Received '
                        '`{}` instead.'.format(type(module)))

    if prefer_all_attr and req_all_attr:
        raise ValueError('`prefer_all_attr` and `req_all_attr` cannot both be True.')

    _valid_merge_methods = {'soft', 'hard', False, None}
    if is_str(merge_dicts):
        merge_dicts = merge_dicts.lower()
    if merge_dicts not in _valid_merge_methods:
        raise ValueError('`merge_dicts` must be in {}.'.format(_valid_merge_methods))
    # Convert to boolean...
    hard = (merge_dicts == 'hard')

    if skip_levels is None:
        skip_levels = []

    dl = _get_attr_dict_list(module, depth, req_all_attr, prefer_all_attr,
                             exclude_childless_modules, skip_levels, filter_func)
    if merge_dicts:
        dl = merge_list_of_dicts(dl, hard=hard, recurse=True)

    return dl
