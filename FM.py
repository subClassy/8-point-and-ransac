'''
Install opencv:
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import random

parser = ArgumentParser()
parser.add_argument("--UseRANSAC", type=int, default=1)
parser.add_argument("--image1", type=str, default='data/notre_dame_1.jpg')
parser.add_argument("--image2", type=str, default='data/notre_dame_2.jpg')
args = parser.parse_args()

print(args)


def get_svd(matrix):
    rank, _ = matrix.shape
    a_t_a = matrix.T@matrix

    v_eigval, v_eigvec = np.linalg.eigh(a_t_a)
    v_eigvec = np.flip(v_eigvec.T, axis=0)

    v_eigval[::-1].sort()
    v_eigval = np.sqrt(v_eigval)

    u_eigvec = np.zeros((matrix.shape[0], matrix.shape[0]))

    for col in range(rank):
        u_eigvec[:, col] = (1 / v_eigval[col]) * (matrix @ v_eigvec[col].T)

    return u_eigvec, v_eigval, v_eigvec


def my_svd(matrix):
    m, n = matrix.shape

    if m <= n:
        return get_svd(matrix)
    else:
        v_eigvec, v_eigval, u_eigvec = get_svd(matrix.T)
        return u_eigvec.T, v_eigval, v_eigvec.T


def normalize_data(x):
    mvec = x.mean(0)

    dist = []
    for pt in range(len(x)):
        cur_pt = x[pt, :]
        cur_dist = np.sqrt(((cur_pt[0] - mvec[0]) ** 2) + ((cur_pt[1] - mvec[1]) ** 2))
        dist.append(cur_dist)

    mdist = np.mean(dist)

    tr_m = [[np.sqrt(2) / mdist, 0, -mvec[0] * (np.sqrt(2) / mdist)],
            [0, np.sqrt(2) / mdist, -mvec[1] * (np.sqrt(2) / mdist)],
            [0, 0, 1]]

    tr_m = np.array(tr_m)

    ones = np.ones((len(x), 1))
    aug_x = np.hstack((x, ones))

    x = aug_x@tr_m.T

    return x.T, tr_m


def FM_by_normalized_8_point(pts1, pts2):
    F_prime, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    # comment out the above line of code. 

    pts1, tr_m_1 = normalize_data(pts1)
    pts2, tr_m_2 = normalize_data(pts2)

    no_of_points = len(pts1[0])
    coeff_matrix = np.zeros((no_of_points, 9))

    for pt in range(no_of_points):
        coeff_matrix[pt] = [pts2[0, pt] * pts1[0, pt], pts2[0, pt] * pts1[1, pt], pts2[0, pt],
                            pts2[1, pt] * pts1[0, pt], pts2[1, pt] * pts1[1, pt], pts2[1, pt],
                            pts1[0, pt], pts1[1, pt], 1]

    _, _, v = my_svd(coeff_matrix)

    F = v[-1].reshape(3, 3)

    u, s, v = my_svd(F)
    s[2] = 0
    F = u @ (np.diag(s) @ v)
    F = (np.transpose(tr_m_2) @ F) @ tr_m_1
    F = F / F[2, 2]

    return F


def FM_by_RANSAC(pts1, pts2):
    F_prime, mask_prime = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    # comment out the above line of code.
    threshold = 1
    iterations = 10
    no_of_points = len(pts1)
    best_mask = np.zeros(no_of_points)
    best_inliers = 0
    best_f = np.zeros((3, 3))
    for _ in range(iterations):
        sample = random.sample(range(0, no_of_points), 8)
        sample_pts1 = pts1[sample]
        sample_pts2 = pts2[sample]
        sample_f = FM_by_normalized_8_point(sample_pts1, sample_pts2)
        error_matrix = np.zeros(no_of_points)

        for pts in range(no_of_points):
            epiline = np.dot(sample_f, np.hstack((pts1[pts], [1])))

            x = pts2[pts][0]
            y = pts2[pts][1]
            d = abs(epiline[0] * x + epiline[1] * y + epiline[2]) / np.sqrt(epiline[0] ** 2 + epiline[1] ** 2)
            error_matrix[pts] = d

        mask = np.int8(np.absolute(error_matrix) < threshold)
        maybe_inliers = len(mask.nonzero()[0])

        if maybe_inliers > best_inliers:
            best_inliers = maybe_inliers
            best_mask = mask
            best_f = sample_f

    return best_f, best_mask


img1 = cv2.imread(args.image1, 0)
img2 = cv2.imread(args.image2, 0)

sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.8 * n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

F = None
if args.UseRANSAC:
    F, mask = FM_by_RANSAC(pts1, pts2)
    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
else:
    F = FM_by_normalized_8_point(pts1, pts2)


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


# Find epilines corresponding to points in second image,  and draw the lines on first image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img6)
plt.show()

# Find epilines corresponding to points in first image, and draw the lines on second image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
plt.subplot(121), plt.imshow(img4)
plt.subplot(122), plt.imshow(img3)
plt.show()
