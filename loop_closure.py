import torch
import numpy as np
import pypose as pp


class LoopClosure:
    def __init__(self):
        self.loop_links = []
        self.loop_poses = []

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
            links = np.loadtxt(links_file)
            if len(links.shape) == 1:
                links = links[np.newaxis, :]
            self.loop_links = links[:, :2].astype(int)

            self.loop_poses = np.loadtxt(poses_file)

    def loop_edges(self):
        if len(self.loop_links) > 0:
            return torch.tensor(self.loop_links), pp.SE3(self.loop_poses).to(torch.float32)
        else:
            return None, None
    

if __name__ == '__main__':
    from Datasets.TrajFolderDataset import TrajFolderDataset
    from imu_integrator import IMUModule
    from loop_closure import LoopClosure
    from pvgo import run_pvgo

    dataroot = '../euroc/MH_01_easy/mav0'
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

    loop_closure = LoopClosure()
    loop_closure.read_from_file(dataroot)

    vo_motions = np.loadtxt(f'{resultdir}/{iteration}/vo_motion.txt')
    pgo_poses = np.loadtxt(f'{resultdir}/{iteration}/pgo_pose.txt')
    pgo_vels = np.loadtxt(f'{resultdir}/{iteration}/pgo_vel.txt')
    imu_drots = np.loadtxt(f'{resultdir}/{iteration}/imu_drot.txt')
    imu_dtrans = np.loadtxt(f'{resultdir}/{iteration}/imu_dtrans.txt')
    imu_dvels = np.loadtxt(f'{resultdir}/{iteration}/imu_dvel.txt')
    
    links = torch.tensor(dataset.links)
    dts = torch.tensor(dataset.rgb_dts, dtype=torch.float32)
    pgo_poses = pp.SE3(pgo_poses).to(torch.float32)
    pgo_vels = torch.tensor(pgo_vels, dtype=torch.float32)
    motions = pp.SE3(vo_motions).to(torch.float32)
    imu_drots = pp.SO3(imu_drots).to(torch.float32)
    imu_dtrans = torch.tensor(imu_dtrans, dtype=torch.float32)
    imu_dvels = torch.tensor(imu_dvels, dtype=torch.float32)
    loop_links, loop_motions = loop_closure.loop_edges()

    print('run pvgo')
    _, _, loop_poses, loop_vels, covs = run_pvgo(
        pgo_poses, pgo_vels,
        motions, links, dts,
        imu_drots, imu_dtrans, imu_dvels,
        loop_links, loop_motions,
        device='cpu', radius=1e4,
        loss_weight=loss_weight,
        target=''
    )

    loop_poses = loop_poses.detach().cpu().numpy()
    loop_vels = loop_vels.detach().cpu().numpy()

    np.savetxt(f'{resultdir}/{iteration}/loop_pose.txt', loop_poses)
    np.savetxt(f'{resultdir}/{iteration}/loop_vel.txt', loop_vels)
