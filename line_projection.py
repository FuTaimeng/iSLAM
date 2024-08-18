import cv2
import torch
import numpy as np
import pypose as pp
import matplotlib.pyplot as plt


def plot_pointcloud(ax, points, colors):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], '.', c=col)
    ax.axis('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def is_inside_image_1D(u, width):
    return torch.logical_and(u >= 0, u <= width)

def is_inside_image(uv, width, height):
    if len(uv.shape) == 3:
        return torch.logical_and(is_inside_image_1D(uv[0, ...], width), is_inside_image_1D(uv[1, ...], height))
    else:
        return torch.logical_and(is_inside_image_1D(uv[:, 0, ...], width), is_inside_image_1D(uv[:, 1, ...], height)) 
    

def uv_projection(disp, intrinsic, baseline):
    fx, fy, cx, cy = intrinsic
    height, width = disp.shape[-2:]

    # disp to depth
    disp_clip = torch.where(disp >= 1, disp, 1)
    z = fx*baseline / disp_clip

    # intrinsic matrix
    K = torch.tensor([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=torch.float32).view(3, 3)
    K_inv = torch.linalg.inv(K)

    # build UV map
    u_lin = torch.linspace(0, width-1, width)
    v_lin = torch.linspace(0, height-1, height)
    u, v = torch.meshgrid(u_lin, v_lin, indexing='xy')
    uv = torch.stack([u, v])
    uv1 = torch.stack([u, v, torch.ones_like(u)])

    # back-project to 3D point
    P = z.unsqueeze(-1) * (K_inv.unsqueeze(0).unsqueeze(0) @ uv1.permute(1, 2, 0).unsqueeze(-1)).squeeze()

    return P


def line_projection(disp, lines, intrinsic, baseline):
    fx, fy, cx, cy = intrinsic

    # disp to depth
    disp_clip = torch.where(disp >= 1, disp, 1)
    z = fx*baseline / disp_clip

    # intrinsic matrix
    K = torch.tensor([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=torch.float32).view(3, 3)
    K_inv = torch.linalg.inv(K)

    # back-project to 3D point
    uv1 = torch.cat([lines, torch.ones(lines.shape[:-1]+(1,))], dim=-1)
    depth = z[0, lines[..., 0], lines[..., 1]]
    P = depth.unsqueeze(-1) * (K_inv.unsqueeze(0).unsqueeze(0) @ uv1.unsqueeze(-1)).squeeze(-1)

    return P


intrinsic = [109.0611, 109.0611, 77.1103, 60.2379]
baseline = 0.1101

disp = np.load('train_results/test_euroc/exp_bs=8_lr=3e-6_lw=(4,0.1,2,0.1)/disp/10.npy')
disp = torch.tensor(disp)

rgb = np.load('train_results/test_euroc/exp_bs=8_lr=3e-6_lw=(4,0.1,2,0.1)/rgb/10.npy')
rgb_cv = (rgb * 255).astype(np.uint8).transpose(1, 2, 0)
cv2.imshow('rgb', rgb_cv)
cv2.waitKey(0)

fig = plt.figure('cloud')
ax = fig.add_subplot(projection='3d')

pts = uv_projection(disp, intrinsic, baseline)
pts = pts.numpy().reshape(-1, 3)

rgb_small = cv2.resize(rgb_cv, None, fx=0.25, fy=0.25)
col = (rgb_small.astype(float) / 255).reshape(-1, 3)

plot_pointcloud(ax, pts, col)

plt.show()
