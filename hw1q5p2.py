
import numpy as np
import pytorch3d
import torch

from starter.utils import get_device, get_points_renderer

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
    
    
def sample_torus_to_pointcloud(num_samples=200, device=None):
    
    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)
    
    R = 2
    r = 0.5
    x = (R + r * torch.sin(Theta)) * torch.cos(Phi)
    y = (R + r * torch.sin(Theta)) * torch.sin(Phi)
    z = r * torch.cos(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())
    
    return points, color

def sample_trefoil_knot_to_pointcloud(num_samples=200, device=None):
    if device is None:
        device = get_device()

    Theta = torch.linspace(0, 2*np.pi, num_samples)
    
    x = torch.sin(Theta) + 2 * torch.sin(2 * Theta)
    y = torch.cos(Theta) - 2 * torch.cos(2 * Theta)
    z = - torch.sin(3 * Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())
    
    return points, color

def main():
    torus_pts, torus_color = sample_torus_to_pointcloud(num_samples=200)
    
    render_pointcloud_to_gif(V=torus_pts, rgb=torus_color, gif_path="hw1q5p2_torus.gif", downsample_factor=10)
    
    trefoil_pts, trefoil_color = sample_trefoil_knot_to_pointcloud(num_samples=2000)
    render_pointcloud_to_gif(V = trefoil_pts, rgb = trefoil_color, gif_path="hw1q5p2_trefoil.gif", downsample_factor=1)
    
if __name__ == "__main__":
    main()