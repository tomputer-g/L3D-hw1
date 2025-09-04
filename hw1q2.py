import imageio
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from starter.utils import get_device, get_mesh_renderer

import pytorch3d
import torch
from tqdm import tqdm

def create_gif(images_list: list[np.ndarray], gif_path: Path, FPS=15):
    # images_list is a list of (H,W,3) images
    assert images_list[0].shape[2] == 3
    
    frame_duration_ms = 1000 // FPS
    imageio.mimsave(gif_path, images_list, duration=frame_duration_ms, loop=0)
    
def render_and_save_gif(gif_path:Path, desc: str, V: torch.tensor, F: torch.tensor):
    device = get_device()
    
    image_size=256

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices = V
    faces = F
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor([0.7, 0.7, 1])  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    
    
    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    images_list = []

    for i in tqdm(range(0, 360, 10), desc="Rendering " + desc + "..."):

        theta = np.radians(i)
        c, s = np.cos(theta), np.sin(theta)
        R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]]).unsqueeze(0)

        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=torch.tensor([[0, 0, 3]]), fov=60, device=device
        )
        
        rend = renderer(mesh, cameras=cameras, lights=lights)
        img = rend.cpu().numpy()[0, ..., :3]
            
        img *= 255
        img = img.astype('uint8')
        images_list.append(img)
    
    create_gif(images_list, gif_path)



def main():
    
    V_tetra = torch.tensor([
        [-0.5,-0.5,-0.5],
        [0,1,0],
        [0,0,1],
        [1,0,0]
    ], dtype=torch.float32)

    F_tetra = torch.tensor([
        [0,1,2],
        [1,2,3],
        [0,2,3],
        [0,1,3]
    ], dtype=torch.long)
    render_and_save_gif("hw1q2_tetrahedron.gif", "tetrahedron", V=V_tetra, F=F_tetra)
    
    V_cube = torch.tensor([
        [-0.5,-0.5,-0.5],
        [0.5,-0.5,-0.5],
        [-0.5,0.5,-0.5],
        [0.5,0.5,-0.5],
        
        [-0.5,-0.5,0.5],
        [0.5,-0.5,0.5],
        [-0.5,0.5,0.5],
        [0.5,0.5,0.5]
    ], dtype=torch.float32)
    
    F_cube = torch.tensor([
        [0,1,2],
        [1,2,3],
        
        [4,5,6],
        [5,6,7],
        
        [0,1,4],
        [1,4,5],
        
        [2,3,6],
        [3,6,7],
        
        [0,2,4],
        [2,4,6],
        
        [1,3,5],
        [3,5,7]
        
    ], dtype=torch.long)
    render_and_save_gif("hw1q2_cube.gif", "cube", V=V_cube, F=F_cube)

if __name__ == "__main__":
    main()