
import numpy as np
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer

from pathlib import Path
import imageio

import mcubes
from tqdm import tqdm

def create_gif(images_list: list[np.ndarray], gif_path: Path, FPS=15):
    # images_list is a list of (H,W,3) images
    assert images_list[0].shape[2] == 3
    
    frame_duration_ms = 1000 // FPS
    imageio.mimsave(gif_path, images_list, duration=frame_duration_ms, loop=0)

def create_torus_mesh(voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.6
    max_value = 1.6
    R = 1
    r = 0.5
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = (torch.sqrt(X**2 + Y**2) - R)**2 + Z**2 - r**2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    return mesh


def create_whatever_mesh(voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -5
    max_value = 5
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = X**4 + Y**4 + Z**4 + 2 * X**2 * Y**2 + 2 * X**2 * Z**2 + 2 * Y**2 * Z**2 + 8*X*Y*Z - 8 * X**2 - 8 * Y**2 - 8 * Z**2 + 15
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    return mesh


def render_mesh_to_gif(
        mesh: pytorch3d.structures.Meshes,
        gif_path: Path,
        image_size=256,
        device=None):
    if device is None:
        device = get_device()

    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)

    image_list = []
    for azimuth in tqdm(range(0, 360, 10), desc="Rendering mesh..."): 
        R, T = pytorch3d.renderer.look_at_view_transform(6, 10, azimuth)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        img = rend.cpu().numpy()[0, ..., :3].clip(0,1)  # (B, H, W, 4) -> (H, W, 3)
        img *= 255
        img = img.astype('uint8')
        image_list.append(img)
        
    create_gif(image_list, gif_path)

def main():
    mesh = create_torus_mesh()
    render_mesh_to_gif(mesh, "hw1q5p3_torus.gif")

    mesh2 = create_whatever_mesh()
    render_mesh_to_gif(mesh2, "hw1q5p3_whatever.gif")

    
if __name__ == "__main__":
    main()