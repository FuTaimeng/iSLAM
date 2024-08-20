import pypose as pp
import numpy as np
import pandas
import torch
import cv2

path = '../SuperGluePretrainedNetwork/dump_match_pairs/1403636581513555456_1403636585563555584_matches.npz'
npz = np.load(path)

print(npz.files)
['keypoints0', 'keypoints1', 'matches', 'match_confidence']
print(npz['keypoints0'].shape)
print(npz['keypoints1'].shape)
print(npz['matches'].shape)
print(np.sum(npz['matches']>-1))
print(npz['match_confidence'].shape)

mask = npz['matches'] != -1
matches = npz['matches'][mask]
points1 = npz['keypoints0'][mask, :]
points2 = npz['keypoints1'][matches, :]
print(points1.shape)
print(points2.shape)

fx, fy, cx, cy = 458.654, 457.296, 367.215, 248.375
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])

E, mask = cv2.findEssentialMat(points1, points2, cameraMatrix1=K, distCoeffs1=dist, cameraMatrix2=K, distCoeffs2=dist, 
                               method=cv2.RANSAC, prob=0.999, threshold=1.0)
mask = mask.astype(bool).squeeze()
inliers1 = points1[mask, :]
inliers2 = points2[mask, :]

x1 = np.concatenate([inliers1, np.ones(len(inliers1)).reshape(-1, 1)], axis=-1).T
x2 = np.concatenate([inliers2, np.ones(len(inliers2)).reshape(-1, 1)], axis=-1).T
inv_K = np.linalg.inv(K)
r = (x2.T @ inv_K.T @ E @ inv_K @ x1).diagonal()
print('r', np.sum(r), r.shape)

_, R, t, _ = cv2.recoverPose(E, inliers1, inliers2, cameraMatrix=K)
print('R', R)
print('t', t)

posefile = 'C:/Users/Tymon/Documents/Projects/euroc/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv'
df = pandas.read_csv(posefile)
timestamp = df.values[:, 0]
poses = df.values[:, (1,2,3, 5,6,7,4)]
vels = df.values[:, 8:11]

index1 = np.where(timestamp == 1403636581513555456)
index2 = np.where(timestamp == 1403636585563555584)
print('index', index1, index2)

T_BC = np.array([0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
         0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
         0.0, 0.0, 0.0, 1.0], dtype=np.float32).reshape(4, 4)
T_BC = pp.from_matrix(T_BC, pp.SE3_type)

gt_poses = pp.SE3(poses) @ T_BC
gt_pose = gt_poses[index2].Inv() @ gt_poses[index1]
print('gt pose', gt_pose)

est_rot = pp.from_matrix(R, ltype=pp.SO3_type).to(torch.float32)
est_trans = torch.tensor(t, dtype=torch.float32).squeeze()
est_trans = est_trans / est_trans.norm() * gt_pose.translation().norm()
est_pose = pp.SE3(torch.cat([est_trans, est_rot.tensor()]))
print('pose', est_pose)

rot_err = (gt_pose.rotation().Inv() @ est_pose.rotation()).Log().norm() * 180/torch.pi
trans_err = (gt_pose.translation() - est_pose.translation()).norm()
print('errs', rot_err, trans_err)
print('dist', gt_pose.translation().norm())

img1 = cv2.imread('C:/Users/Tymon/Documents/Projects/euroc/MH_01_easy/mav0/cam0/data/1403636581513555456.png')
img2 = cv2.imread('C:/Users/Tymon/Documents/Projects/euroc/MH_01_easy/mav0/cam0/data/1403636585563555584.png')
img_matches = cv2.hconcat([img1, img2])
for i in range(0, len(inliers1), 10):
    p1 = inliers1[i].astype(int)
    p2 = inliers2[i].astype(int)
    p2[0] += img1.shape[1]
    cv2.line(img_matches, tuple(p1), tuple(p2), color=(0,0,255))
cv2.imshow('matches', img_matches)

# # Initialize the ORB detector
# orb = cv2.ORB_create()

# # Find the keypoints and descriptors with ORB
# keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
# keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# # Initialize the matcher
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# # Match descriptors
# matches = bf.match(descriptors1, descriptors2)

# # Sort matches based on their distance (i.e., quality of match)
# matches = sorted(matches, key=lambda x: x.distance)

# img_matches2 = cv2.hconcat([img1, img2])
# for i in range(0, len(matches), 10):
#     p1 = keypoints1[matches[i].trainIdx].pt
#     p2 = keypoints2[matches[i].queryIdx].pt
#     p1 = (int(p1[0]), int(p1[1]))
#     p2 = (int(p2[0]+img1.shape[1]), int(p2[1]))
#     cv2.line(img_matches2, p1, p2, color=(0,0,255))
# cv2.imshow('matches2', img_matches2)

def skew_symmetric(v):
    v = v.reshape(-1)
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

E2 = skew_symmetric(t) @ R
E3 = skew_symmetric(gt_pose.translation().numpy()) @ gt_pose.rotation().matrix().squeeze().numpy()
E2 = E2 / np.sum(E2) * np.sum(E)
E3 = E3 / np.sum(E3) * np.sum(E)
print(E)
print(E2)
print(E3)

cv2.waitKey(0)