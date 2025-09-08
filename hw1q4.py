"""
Usage:
    python -m starter.camera_transforms --image_size 512
"""

import matplotlib.pyplot as plt
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer


def render_textured_cow(
    cow_path="data/cow.obj",
    image_size=256,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)
    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative
    renderer = get_mesh_renderer(image_size=256)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()


if __name__ == "__main__":
    Rs = [
        [[1,0,0],[0,1,0],[0,0,1]], #identity
        [[0,1,0],[-1,0,0],[0,0,1]],#RotZ_cw_90
        [[1,0,0],[0,1,0],[0,0,1]],
        [[1,0,0],[0,1,0],[0,0,1]], 
        [[0,0,1],[0,1,0],[-1,0,0]],#RotY_cw_90
    ]
    
    Ts = [
        [0,0,0],
        [0,0,0],
        [0,0,2], #zoom out
        [0.5,-0.5,0], #move bottom left
        [-3,0,3], #reset cam after rotation and go to z=+3
    ]
    
    jpg_names = [
        "identity",
        "q1_rotz_cw_90",
        "q2_zoom_out",
        "q3_move_bottom_left",
        "q4_roty_cw_90",
    ]
    
    for i in range(len(Rs)):
        img = render_textured_cow(R_relative=Rs[i], T_relative=Ts[i])
        plt.imsave("hw1q4_"+jpg_names[i]+".jpg", img)
