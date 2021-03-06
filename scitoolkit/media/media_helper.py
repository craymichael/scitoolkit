# =====================================================================
# media_helper.py - A scitoolkit file
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
import cv2
import re
import os
from skimage.transform import resize as imresize

from scitoolkit.system.sys_helper import sys_has_display

CS_REGEX = re.compile(r'COLOR_RGB2([A-Z]+)$')
ALL_CS = ['RGB', 'GAUSSIAN']
# Find OpenCV version
CV2_MAJOR_VER, CV2_MINOR_VER, CV2_SUBMINOR_VER = map(int, cv2.__version__.split('.')[:3])


def _gen_valid_colorspaces():
    cs_groups = [CS_REGEX.findall(attr) for attr in dir(cv2)]
    return [cs[0] for cs in cs_groups if len(cs)]


# Extend with all available color spaces
ALL_CS.extend(_gen_valid_colorspaces())
# Base string used in cv2 colorspace conversions
CS_STR = 'COLOR_{}2{}'


def get_valid_colorspaces():
    return ALL_CS.copy()


def cvt2gaussian(im, cs_orig='RGB'):
    if cs_orig.upper() == 'RGB':
        assert len(im.shape) == im.shape[-1] == 3
        # normalize to [0,1]
        im = im.astype(np.float32) / 255
        shp = im.shape
        im = im.reshape(-1, 3).T
        gim = np.linalg.inv(np.asarray([
            [0.96, 0., 0.],
            [0., 0.69, 0.],
            [0., 0., 1.11]
        ])).dot(
            np.asarray([
                [0.06, 0.63, 0.27],
                [0.30, 0.04, -0.35],
                [0.34, -0.60, 0.17]
            ]).dot(im) + np.asarray([
                [0.], [0.35], [0.6]
            ])
        )
        gim = gim.T.reshape(shp)
        return gim
    else:
        raise NotImplementedError('Colorspace {} is not '
                                  'implemented.'.format(cs_orig))


def get_colorspace_conversion_str(from_cs, to_cs):
    from_cs = from_cs.upper()
    to_cs = to_cs.upper()
    for cs in (from_cs, to_cs):
        if to_cs not in ALL_CS:
            raise ValueError('`{}` is not a valid colorspace. Valid '
                             'colorspaces:\n{}'.format(cs, ALL_CS))
    return CS_STR.format(from_cs, to_cs)


def convert_colorspace(im, from_cs, to_cs):
    if to_cs.upper() == 'GAUSSIAN':
        return cvt2gaussian(im, cs_orig=from_cs)
    else:
        return cv2.cvtColor(im, get_colorspace_conversion_str(from_cs, to_cs))


def get_image(image_path, cs='RGB', resize=None):
    """"""
    if image_path.split('.')[-1].lower() == 'gif':  # Assumes 1-frame gif
        im = cv2.VideoCapture(image_path).read()[1]
    else:
        # Images loaded as height by width (y, x)
        im = cv2.imread(image_path)

    im = convert_colorspace(im, 'BGR', cs)

    if resize is not None:
        im = resize_image(im, resize)
    return im


def resize_image(im, resize):
    if not isinstance(resize, (list, tuple, np.ndarray)):
        # Assume float/scaling factor
        h, w = im.shape[:2]
        h = max(round(h * resize), 1)
        w = max(round(w * resize), 1)
        resize = [h, w]
    im = imresize(im, (resize, resize), mode='constant',
                  preserve_range=True, order=1)
    return im


def lin_contrast_stretch(im, px_max=255):
    """"""
    dtype = im.dtype
    im = im.astype(float)
    return ((im - np.min(im)) /
            (np.max(im) - np.min(im)) * px_max).astype(dtype)


def get_fps(vid_cap):
    if CV2_MAJOR_VER < 3:
        fps = vid_cap.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
    return fps


def play_video(frames, fps=None, dur=None, out_file=None, fourcc_code='MJPG'):
    assert fps is not None or dur is not None, 'Either "fps" or "dur" must be specified'

    h, w = frames.shape[1:3]
    if fps is not None:
        wait_dur = int(1000. / fps)  # ms
    else:
        wait_dur = int(1000. * dur / len(frames))
        fps = int(1000. / wait_dur)

    if out_file:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        out = cv2.VideoWriter(out_file, fourcc, fps, (w, h),
                              isColor=int(len(frames.shape) == 4 and
                                          frames.shape[-1] == 3))
    else:
        out = None

    for frame in frames:
        if out_file:
            out.write(frame)
        if 'DISPLAY' in os.environ.keys():
            cv2.imshow('frame', frame)
        if cv2.waitKey(wait_dur) & 0xFF == ord('q'):
            break

    if out_file:
        out.release()
    cv2.destroyAllWindows()


def read_video(file_name, resize=None):
    vid_cap = cv2.VideoCapture(file_name)
    frames = []
    fps = get_fps(vid_cap)

    while vid_cap.isOpened():
        ret, frame = vid_cap.read()

        if not ret:
            break

        if resize is not None:
            frame = resize_image(frame, resize)

        frames.append(frame)

    vid_cap.release()
    frames = np.asarray(frames)
    # print('Read in {} frames.'.format(len(frames)))
    return frames, fps
