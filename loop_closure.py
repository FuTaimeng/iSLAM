import bisect

import torch
import numpy as np
import pypose as pp


class LoopClosure:
    def __init__(self):
        self.loop_links = []
        self.loop_poses = []
        self.keyframes = []
        self.poses = []
        self.num_frames= 0

        # euroc
        self.interval_threshold = 10
        self.rotation_threshold = np.deg2rad(5)
        self.translation_threshold = 0.2

    def test_keyframe(self, i):
        index = bisect.bisect(self.keyframes, i) - 1
        if index < 0:
            return True
        j = self.keyframes[index]
        if i - j >= self.interval_threshold:
            return True
        motion = self.poses[i].Inv() @ self.poses[j]
        if motion.translation().norm() >= self.translation_threshold:
            return True
        if motion.rotation().Log().norm() >= self.rotation_threshold:
            return True
        return False

    def add_frames(self, poses):
        self.poses.extend(poses)
        for i in range(self.num_frames, self.num_frames + len(poses)):
            if self.test_keyframe(i) and i not in self.keyframes:
                bisect.insort(self.keyframes, i)

    def add_loop(self, i, j, motion):
        self.loop_links.append([i, j])
        self.loop_poses.append(motion)
        if i not in self.keyframes:
            bisect.insort(self.keyframes, i)
        if j not in self.keyframes:
            bisect.insort(self.keyframes, j)

    def get_loopedges(self):
        if len(self.loop_links) > 0:
            return torch.tensor(self.loop_links), pp.SE3(np.stack(self.loop_poses)).to(torch.float32)
        else:
            return None, None
        
    def get_keyframes(self):
        return torch.tensor(self.keyframes)

    # for testing
    def read_from_file(self, data_dir):
        looproot = '../SuperGluePretrainedNetwork/assets/loop_edges1-5'
        datadir2loopdir = {
            'MH_01_easy': 'result_euroc-MH01',
            '2011_09_30_drive_0018_sync': 'result_kitti05'
        }
        flag = False
        for k, v in datadir2loopdir.items():
            if k in data_dir:
                links_file = f'{looproot}/{v}/loop_final.txt'
                poses_file = f'{looproot}/{v}/loop_final_motion.txt'
                flag = True
                break

        if flag:
            links = np.loadtxt(links_file, dtype=int)
            poses = np.loadtxt(poses_file, dtype=np.float32)
            if len(links.shape) == 1:
                links = links[np.newaxis, :]
                poses = poses[np.newaxis, :]

            for i in range(len(links)):
                self.add_loop(links[i, 0], links[i, 1], poses[i])
    

if __name__ == '__main__':
    from Datasets.TrajFolderDataset import TrajFolderDataset
    from imu_integrator import IMUModule
    from loop_closure import LoopClosure
    from pvgo import run_pvgo

    dataroot = '/data/euroc/MH_01_easy/mav0'
    datatype = 'euroc'
    imu_denoise_model_name = './models/1029_euroc_no_cov_1layer_epoch_100_train_loss_0.19208121810994155.pth'
    resultdir = 'train_results/test_euroc/exp_bs=8_lr=3e-6_lw=(4,0.1,2,0.1)'
    iteration = '1'
    loss_weight = (4,0.1,2,0.1, 4)

    print('load dataset')
    dataset = TrajFolderDataset(
        datadir=dataroot, datatype=datatype, transform=None,
        start_frame=0, end_frame=-1
    )

    imu_module = IMUModule(
        dataset.accels, dataset.gyros, dataset.imu_dts,
        dataset.accel_bias, dataset.gyro_bias,
        dataset.imu_init, dataset.gravity, dataset.rgb2imu_sync, 
        device='cuda', denoise_model_name=imu_denoise_model_name,
        denoise_accel=True, denoise_gyro=(dataset.datatype!='kitti')
    )

    loop_closure = LoopClosure()
    loop_closure.read_from_file(dataroot)

    vo_motions = np.loadtxt(f'{resultdir}/{iteration}/vo_motion.txt')
    pgo_poses = np.loadtxt(f'{resultdir}/{iteration}/pgo_pose.txt')
    pgo_vels = np.loadtxt(f'{resultdir}/{iteration}/pgo_vel.txt')
    # imu_drots = np.loadtxt(f'{resultdir}/{iteration}/imu_drot.txt')
    # imu_dtrans = np.loadtxt(f'{resultdir}/{iteration}/imu_dtrans.txt')
    # imu_dvels = np.loadtxt(f'{resultdir}/{iteration}/imu_dvel.txt')

    pgo_poses = pp.SE3(pgo_poses).to(torch.float32)
    pgo_vels = torch.tensor(pgo_vels, dtype=torch.float32)
    motions = pp.SE3(vo_motions).to(torch.float32)

    # print(motions.rotation().Log().norm(dim=-1).mean() * 180/3.14)
    # print(motions.translation().norm(dim=-1).mean())

    loop_closure.add_frames(pgo_poses)
    keyframes = loop_closure.get_keyframes()
    print('keyframes', len(keyframes))

    print('imu integration')
    imu_dtrans_list = []
    imu_drots_list = []
    imu_dvels_list = []
    for i in range(len(keyframes)-1):
        st = keyframes[i]
        end = keyframes[i+1]
        init_state = {'rot':pgo_poses[st].rotation().numpy(), 'pos':pgo_poses[st].translation().numpy(), 'vel':pgo_vels[st].numpy()}
        imu_dtrans, imu_drots, imu_dcovs, imu_dvels = imu_module.integrate(
            st, end, init_state, motion_mode=True, batch_mode=True
        )
        imu_dtrans_list.extend(imu_dtrans.detach().cpu().numpy())
        imu_drots_list.extend(imu_drots.detach().cpu().numpy())
        imu_dvels_list.extend(imu_dvels.detach().cpu().numpy())

    dts = torch.tensor(dataset.rgb_dts, dtype=torch.float32)
    # imu_drots = pp.SO3(imu_drots).to(torch.float32)
    # imu_dtrans = torch.tensor(imu_dtrans, dtype=torch.float32)
    # imu_dvels = torch.tensor(imu_dvels, dtype=torch.float32)
    imu_drots = pp.SO3(np.stack(imu_drots_list))
    imu_dtrans = torch.tensor(np.stack(imu_dtrans_list), dtype=torch.float32)
    imu_dvels = torch.tensor(np.stack(imu_dvels_list), dtype=torch.float32)
    loop_links, loop_motions = loop_closure.get_loopedges()

    pgo_poses = pgo_poses[keyframes]
    pgo_vels = pgo_vels[keyframes]
    motions_list = []
    for i in range(len(keyframes)-1):
        st = keyframes[i]
        end = keyframes[i+1]
        m = motions[st]
        for j in range(st+1, end):
            m = motions[j] @ m
        motions_list.append(m)
    motions = pp.SE3(torch.stack(motions_list))
    links = torch.tensor([[i, i+1] for i in range(len(keyframes)-1)])
    dts_list = []
    for i in range(len(keyframes)-1):
        st = keyframes[i]
        end = keyframes[i+1]
        dts_list.append(torch.sum(dts[st:end]))
    dts = torch.stack(dts_list)
    loop_links = [[torch.where(keyframes==l[0])[0].item(), torch.where(keyframes==l[1])[0].item()] for l in loop_links]
    loop_links = torch.tensor(loop_links)

    print('run pvgo')
    _, _, loop_poses, loop_vels, covs = run_pvgo(
        pgo_poses, pgo_vels,
        motions, links, dts,
        imu_drots, imu_dtrans, imu_dvels,
        loop_links, loop_motions,
        device='cuda', radius=1e4,
        loss_weight=loss_weight,
        target=''
    )

    loop_poses = loop_poses.detach().cpu().numpy()
    loop_vels = loop_vels.detach().cpu().numpy()

    np.savetxt(f'{resultdir}/{iteration}/loop_pose_{loss_weight[4]}.txt', loop_poses)
    np.savetxt(f'{resultdir}/{iteration}/loop_vel_{loss_weight[4]}.txt', loop_vels)
