'''
Install opencv:
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--UseRANSAC", type=int, default=0)
parser.add_argument("--image1", type=str, default='data/myleft.jpg')
parser.add_argument("--image2", type=str, default='data/myright.jpg')
args = parser.parse_args()

print(args)


def normalize_data(x):
    mvec = x.mean(0)

    dist = []
    for pt in range(len(x)):
        cur_pt = x[pt, :]
        cur_dist = np.sqrt(((cur_pt[0] - mvec[0]) ** 2) + ((cur_pt[1] - mvec[1]) ** 2))
        dist.append(cur_dist)

    mdist = cur_dist.mean()

    tr_m = [[np.sqrt(2) / mdist, 0, -mvec[0] * (np.sqrt(2) / mdist)],
            [0, np.sqrt(2) / mdist, -mvec[1] * (np.sqrt(2) / mdist)],
            [0, 0, 1]]

    ones = np.ones((len(x), 1))
    aug_x = np.hstack((x, ones))

    x = aug_x@np.array(tr_m).T

    return x


def FM_by_normalized_8_point(pts1, pts2):
    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    # comment out the above line of code. 

    pts1 = normalize_data(pts1).T
    pts2 = normalize_data(pts2).T

    no_of_points = len(pts1[0])
    coeff_matrix = np.zeros((no_of_points, 9))

    for pt in range(no_of_points):
        # coeff_matrix[pt, 0] = pts1[pt, 0] * pts2[pt, 0]
        # coeff_matrix[pt, 1] = pts1[pt, 0] * pts2[pt, 1]
        # coeff_matrix[pt, 2] = pts1[pt, 0]
        # coeff_matrix[pt, 3] = pts1[pt, 1] * pts2[pt, 0]
        # coeff_matrix[pt, 4] = pts1[pt, 1] * pts2[pt, 1]
        # coeff_matrix[pt, 5] = pts1[pt, 1]
        # coeff_matrix[pt, 6] = pts2[pt, 0]
        # coeff_matrix[pt, 7] = pts2[pt, 1]
        # coeff_matrix[pt, 8] = 1
        coeff_matrix[i] = [pts1[0, i] * pts2[0, i], pts1[0, i] * pts2[1, i], pts1[0, i] * pts2[2, i],
                           pts1[1, i] * pts2[0, i], pts1[1, i] * pts2[1, i], pts1[1, i] * pts2[2, i],
                           pts1[2, i] * pts2[0, i], pts1[2, i] * pts2[1, i], pts1[2, i] * pts2[2, i]]

    a_t_a = coeff_matrix.T @ coeff_matrix
    _, s, v = np.linalg.svd(a_t_a)

    F_prime = v[-1].reshape(3, 3)

    U, S, V = np.linalg.svd(F_prime)
    S[2] = 0
    F_prime = np.dot(U, np.dot(np.diag(S), V))
    F_prime = F_prime / F_prime[2, 2]

    # F:  fundmental matrix
    return F


def FM_by_RANSAC(pts1, pts2):
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    # comment out the above line of code. 

    # Your task is to implement the algorithm by yourself.
    # Do NOT copy&paste any online implementation. 

    # F:  fundmental matrix
    # mask:   whetheter the points are inliers
    return F, mask


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
