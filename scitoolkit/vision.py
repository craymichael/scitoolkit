# =====================================================================
# vision.py - A scitoolkit file
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

import cv2
import numpy as np


def SIFTSSD(ima, imb, keep=None):  # TODO split into 2+ functions
    sift = cv2.xfeatures2d.SIFT_create()
    # A
    acorners, afeats = sift.detectAndCompute(ima, None)
    # B
    bcorners, bfeats = sift.detectAndCompute(imb, None)
    # Convert corners
    acorners = np.asarray([[kp.pt[1], kp.pt[0]] for kp in acorners])  # no T now...
    bcorners = np.asarray([[kp.pt[1], kp.pt[0]] for kp in bcorners]).T

    # Compute SSDs  # TODO reimplement for generic distance (see metrics...)
    besta2b_raw = np.zeros((len(afeats)))  # ,len(bfeats)))
    dists = np.zeros_like(besta2b_raw)
    for i, f in enumerate(afeats):
        ssds = np.sum((f - bfeats) ** 2, axis=1)
        min_idx = np.argmin(ssds)
        besta2b_raw[i] = min_idx
        dists[i] = ssds[min_idx]

    if keep is not None:
        # Only keep k lowest
        low_idx = np.argpartition(dists, keep)[:keep]
        besta2b = besta2b_raw[low_idx]
        # Fix keypoints
        acorners = acorners[low_idx]
    else:
        besta2b = besta2b_raw
    acorners = acorners.T
    return besta2b, acorners, bcorners


def four_point_algorithm(a, b):
    # Setup A matrix, b vector
    A = []
    B = []
    for i in range(len(a)):
        lhs = [a[i][0], a[i][1], 1]
        rowe = lhs + [0, 0, 0, -a[i][0] * b[i][0], -a[i][1] * b[i][0]]
        rowo = [0, 0, 0] + lhs + [-a[i][0] * b[i][1], -a[i][1] * b[i][1]]
        B.extend(b[i][:2])
        A.extend([rowe, rowo])

    A = np.asarray(A)
    B = np.asarray(B)

    # Compute H
    H = np.linalg.pinv(A).dot(B)

    return H


def ransac_homography(points_a, points_b, iter, thr, inlier_ratio):
    """Computes a homography using RANSAC"""
    # Ensure 2D
    if points_a.shape[1] != 2 or points_b.shape[1] != 2:
        raise ValueError('`points_a` and `points_b` must have 2 dimensions '
                         'only.')
    if points_a.shape != points_b.shape:
        raise ValueError('`points_a` and `points_b` must have the same '
                         'shape.')

    n_points = points_a.shape[0]
    xa = points_a[:, 0]
    ya = points_a[:, 1]
    xb = points_b[:, 0]
    yb = points_b[:, 1]
    # "best" placeholder parameters
    best_n_inliers = -1
    best_dm = None
    best_inliers = None
    best_H = None
    best_coords = None

    def _fix_h(_h):
        return np.append(_h, 1).reshape(3, 3)

    # Main loop
    for i in range(iter):
        # Sample from scitoolkit.oints (min number to fit model)
        samp_idx = np.random.choice(n_points, size=4)
        sampa = points_a[samp_idx]
        sampb = points_b[samp_idx]
        # Fit sample
        H = four_point_algorithm(sampa, sampb)
        H = _fix_h(H)
        # Q6 says this is OK
        pb_H = cv2.perspectiveTransform(np.asarray([points_a]), H)
        pb_H = pb_H[0]
        # print(pb_H)
        # print(points_b)
        # Compute number of inliers
        distance = np.sqrt((pb_H[:, 0] - xb) ** 2 + (pb_H[:, 1] - yb) ** 2)
        inliers = distance <= thr
        # Check if best fit so far
        n_inliers = inliers.sum()
        if n_inliers / n_points >= inlier_ratio and \
                n_inliers >= best_n_inliers:
            # Compare best average distances
            dm = distance.mean()
            if best_dm is None or dm < best_dm:
                best_dm = dm
                best_H = four_point_algorithm(points_a[inliers],
                                              points_b[inliers])
                best_H = _fix_h(best_H)
                best_inliers = inliers
                best_n_inliers = n_inliers
                # Compute closest correspondence
                close_idx = distance.argmin()
                # best_coord = points_a[close_idx]
                # best_coord = cv2.perspectiveTransform(np.asarray([[best_coord]]), H)
                best_coord = points_b[close_idx]
                best_coord = cv2.perspectiveTransform(np.asarray([[best_coord]]), np.linalg.pinv(H))
                # best_coord = warp_coords(np.asarray([best_coord]), np.linalg.pinv(H))

                # best_coords = (best_coord[0,0], points_b[close_idx])
                best_coords = (points_a[close_idx], points_b[close_idx])
    # Wrap things up
    if best_n_inliers == -1:
        raise RuntimeError('Failed to find H using RANSAC with given '
                           'parameters.')
    else:
        print('Best fit has {} inliers.'.format(best_n_inliers))
    return best_H, best_coords  # , best_inliers


def stitch_images(images):
    """Automatically stitches images together"""
    for i, im in enumerate(images):  # TODO this is terrible
        padh = im.shape[0]
        padw = im.shape[1]
        # PAD THEM
        if len(im.shape) == 3:  # color channels, probably
            im = np.pad(im, mode='constant', pad_width=((padh, padh),
                                                        (padw, padw),
                                                        (0, 0)))
        else:
            im = np.pad(im, mode='constant', pad_width=((padh, padh),
                                                        (padw, padw)))
        images[i] = im

    n_ims = len(images)
    im_out_running = images[0]
    for i in range(1, n_ims):
        # Get two images in question
        im1 = images[i]
        # im2 = images[i+1]
        # Compute keypoints
        besta2b, acorners, bcorners = SIFTSSD(im1, im_out_running.astype(np.uint8), keep=200)
        # Create correspondences
        points_a = acorners.T
        points_b = bcorners.T[besta2b.astype(int)]
        points_a = points_a[:, [1, 0]]
        points_b = points_b[:, [1, 0]]

        # Use RANSAC to compute homography
        H, coords = ransac_homography(points_a, points_b, iter=500, thr=20,
                                      inlier_ratio=0.2)
        print('HOMO\n', H)
        # Apply homography
        im1_h = cv2.warpPerspective(im1, H, (im1.shape[1], im1.shape[0]))

        # Calculate image placement
        yp, xp = im_out_running.shape[:2]  # prev image shape
        yn, xn = im1.shape[:2]  # new image shape
        ac, bc = coords
        acx, acy = ac.astype(int)  # new image keypoint coords
        bcx, bcy = bc.astype(int)  # prev image keypoint coords
        # acy, acx = ac.astype(int)#new image keypoint coords
        # bcy, bcx = bc.astype(int)#prev image keypoint coords
        print(coords)

        # plt.figure()
        # plt.imshow(im1, cmap='gray')
        # plt.plot(acx, acy, 'ro', markersize=7)
        # plt.title('HOMO im1 shape')
        # plt.show(block=False)

        cc = ac
        cc = cv2.perspectiveTransform(np.asarray([[ac]]), np.linalg.pinv(H))[0, 0]
        acx, acy = bcx, bcy

        # best_coords = (best_coord[0,0], points_b[close_idx])
        # plt.figure()
        # plt.imshow(im1_h, cmap='gray')
        # plt.plot(cc[0], cc[1], 'ro', markersize=7)
        # plt.plot(acx, acy, 'bo', markersize=7)
        # plt.plot(bcx, bcy, 'bo', markersize=7)
        # plt.title('HOMO im1_h shape')
        # plt.show(block=False)

        # Compute y left overhang
        oyl = np.maximum(acy - bcy, 0)
        # Compute y right overhang
        oyr = np.maximum((yn - acy) + bcy - yp, 0)
        # Compute x top overhang
        oxt = np.maximum(acx - bcx, 0)
        # Compute x bottom overhang
        oxb = np.maximum((xn - acx) + bcx - xp, 0)
        # Output image size
        # print(oyl, oyr, oxt, oxb)
        im_out = np.zeros((yp + oyl + oyr,
                           xp + oxt + oxb))
        # Insert new image
        # im_out[oyl+bcy-acy:oyl+bcy-acy+yn,
        #       oxt+bcx-acx:oxt+bcx-acx+xn] = im1_h
        im_out = cv2.warpPerspective(im1, H, (im_out.shape[1], im_out.shape[0]))
        # plt.figure()
        # plt.imshow(im_out, cmap='gray')
        # plt.title('HOMOE im_out shape')
        # plt.show(block=False)

        # plt.figure()
        # plt.imshow(im_out_running, cmap='gray')
        # plt.plot(bcx, bcy, 'ro', markersize=7)
        # plt.title('Running image')
        # plt.show(block=False)

        # im_out[oyl+bcy-acy:oyl+bcy-acy+yn,
        #       oxt+bcx-acx-200:oxt+bcx-acx+xn-200] = im1_h

        # Cover with old image
        # im_out[oyl:oyl+yp,oxt:oxt+xp] = im_out_running
        # where_zero = np.logical_and(im_out[oyl:oyl+yp,oxt:oxt+xp]==0, im_out_running==0)
        where_zero = im_out[oyl:oyl + yp, oxt:oxt + xp] == 0  # im_out_running==0)
        im_out = im_out.astype(float)
        im_out_running = im_out_running.astype(float)
        im_out[oyl:oyl + yp, oxt:oxt + xp] = (im_out[oyl:oyl + yp, oxt:oxt + xp] + im_out_running)
        im_out[oyl:oyl + yp, oxt:oxt + xp][~where_zero] = (im_out[oyl:oyl + yp, oxt:oxt + xp][~where_zero] / 2)
        im_out_running = im_out.astype(np.uint8)
    return im_out_running
