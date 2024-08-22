import numpy as np
from matplotlib import pyplot as plt

# result_name = 'test_kitti_g'
# exp_name = 'exp_bs=8_lr=3e-6_lw=(1,0.1,10,0.1)'
result_name = 'test_euroc'
exp_name = 'exp_bs=8_lr=3e-6_lw=(4,0.1,2,0.1)'
# result_name = 'test_tartanair_g'
# exp_name = 'exp_bs=8_lr=3e-6_lw=(1.5,0.125,1.6875,0.025)_'
gt_poses = np.loadtxt(f'../train_results/{result_name}/{exp_name}/gt_pose.txt')
vo_poses = np.loadtxt(f'../train_results/{result_name}/{exp_name}/1/vo_pose.txt')
pgo_poses = np.loadtxt(f'../train_results/{result_name}/{exp_name}/1/pgo_pose.txt')
imu_poses = np.loadtxt(f'../train_results/{result_name}/{exp_name}/1/imu_pose.txt')
loop_poses = np.loadtxt(f'../train_results/{result_name}/{exp_name}/1/loop_pose_4.txt')

fig, ax = plt.subplots()
ax.plot(vo_poses[:, 0], vo_poses[:, 1])
ax.plot(pgo_poses[:, 0], pgo_poses[:, 1])
ax.plot(gt_poses[:, 0], gt_poses[:, 1])
# ax.plot(imu_poses[:, 0], imu_poses[:, 1])
ax.plot(loop_poses[:, 0], loop_poses[:, 1])

ax.legend(['vo', 'pgo', 'gt', 'loop'])

ax.axis('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')

fig.savefig(f'traj_{result_name}_{exp_name}.png')
