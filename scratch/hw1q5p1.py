import numpy as np
import pytorch3d
import torch
from starter.utils import get_device, get_points_renderer, unproject_depth_image
from starter.render_generic import load_rgbd_data

from pathlib import Path
import imageio

from tqdm import tqdm


def create_gif(images_list: list[np.ndarray], gif_path: Path, FPS=15):
    # images_list is a list of (H,W,3) images
    assert images_list[0].shape[2] == 3
    
    frame_duration_ms = 1000 // FPS
    imageio.mimsave(gif_path, images_list, duration=frame_duration_ms, loop=0)


def render_pointcloud_to_gif(
    V: torch.Tensor,
    rgb: torch.Tensor,
    gif_path: Path,
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
    downsample_factor=1,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    
    verts = V[::downsample_factor].to(device).unsqueeze(0)
    rgb = rgb[::downsample_factor].to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    
    image_list = []
    for azimuth in tqdm(range(0, 360, 10), desc="Rendering pointcloud..."): 
        R, T = pytorch3d.renderer.look_at_view_transform(6, 10, azimuth)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(point_cloud, cameras=cameras)
        img = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        img *= 255
        img = img.astype('uint8')
        image_list.append(img)
        
    create_gif(image_list, gif_path)

def main():
    data = load_rgbd_data() #dict_keys(['rgb1', 'mask1', 'depth1', 'rgb2', 'mask2', 'depth2', 'cameras1', 'cameras2'])
    pc1_points, pc1_rgb = unproject_depth_image(image=torch.Tensor(data['rgb1']), mask=torch.Tensor(data['mask1']), depth=torch.Tensor(data['depth1']), camera=data['cameras1'])
    pc2_points, pc2_rgb = unproject_depth_image(image=torch.Tensor(data['rgb2']), mask=torch.Tensor(data['mask2']), depth=torch.Tensor(data['depth2']), camera=data['cameras2'])
    # print(pc1_points.shape) #125035,3
    # print(pc1_rgb.shape)    #125035,4
    union_points = torch.vstack([pc1_points,pc2_points])
    union_rgb = torch.vstack([pc1_rgb, pc2_rgb])
    
    render_pointcloud_to_gif(V=pc1_points, rgb=pc1_rgb, gif_path="hw1q5p1_pc1.gif", downsample_factor=10)
    render_pointcloud_to_gif(V=pc2_points, rgb=pc2_rgb, gif_path="hw1q5p1_pc2.gif", downsample_factor=10)
    render_pointcloud_to_gif(V=union_points, rgb=union_rgb, gif_path="hw1q5p1_pc_both.gif", downsample_factor=10)



if __name__ == "__main__":
    main()