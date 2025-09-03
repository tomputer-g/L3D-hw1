import imageio
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from starter.utils import get_device, get_mesh_renderer, load_cow_mesh

import pytorch3d
import torch
from tqdm import tqdm
def create_gif(images_list: list[np.ndarray], gif_path: Path, FPS=15):
    # images_list is a list of (H,W,3) images
    assert images_list[0].shape[2] == 3
    
    frame_duration_ms = 1000 // FPS
    imageio.mimsave(gif_path, images_list, duration=frame_duration_ms, loop=0)

def main():
    device = get_device()
    cow_path = "data/cow.obj"
    image_size=256

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
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

    for i in tqdm(range(360), desc="Rendering cow..."):

        theta = np.radians(i)
        c, s = np.cos(theta), np.sin(theta)
        R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]]).unsqueeze(0)

        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=torch.tensor([[0, 0, 3]]), fov=60, device=device
        )


        rend = renderer(mesh, cameras=cameras, lights=lights)
        img = rend.cpu().numpy()[0, ..., :3]
            
        img *= 256
        img = img.astype('uint8')
        images_list.append(img)
    
    # print(img.max())
    # print(img.min()) #this is fine but background is black instead of white
    
    # img = imageio.imread(Path("images/cow_render.jpg"))
    # plt.imsave('out_cow.jpg', img)

    # print("Image shape", img.shape) #256, 256, 3
    # print("Image type", img.dtype) #uint8
    # images_list = [img] * 10
    create_gif(images_list, Path('out.gif'))

if __name__ == "__main__":
    main()