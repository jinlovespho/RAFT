import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

from torchvision.utils import save_image 


DEVICE = 'cuda'


import torch.nn.functional as F

def warp_image(image, flow):
    """
    Warp `image` using the backward optical flow `flow`.
    
    image: (1, 3, H, W) - source image (e.g., frame1)
    flow:  (1, 2, H, W) - flow field from frame1 → frame2

    Returns:
        Warped image (1, 3, H, W), aligned with the target frame.
    """
    N, C, H, W = image.shape
    # Generate a mesh grid of pixel coordinates (x, y)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=image.device),
        torch.arange(W, device=image.device),
        indexing='ij'
    )

    # Stack into (2, H, W) and add batch dimension → (1, 2, H, W)
    base_grid = torch.stack((grid_x, grid_y), dim=0).float()  # (2, H, W)
    base_grid = base_grid.unsqueeze(0)  # (1, 2, H, W)

    # Add flow to base grid to get target sampling locations
    vgrid = base_grid + flow  # (1, 2, H, W)

    # Normalize grid values to [-1, 1] for grid_sample
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0  # x
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0  # y

    # Rearrange to (N, H, W, 2) for grid_sample
    vgrid = vgrid.permute(0, 2, 3, 1)

    # Sample pixels from image using flow
    warped = F.grid_sample(image, vgrid, align_corners=True, mode='bilinear', padding_mode='border')
    return warped



def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, img1_id, save_path, concat_flow):
    img = img[0].permute(1,2,0).cpu().numpy()       # H W 3 [0,255]
    flo = flo[0].permute(1,2,0).cpu().numpy()       # H W 2 
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)               # H W 3
    
    if concat_flow:
        img_flo = np.concatenate([img, flo], axis=0)    # H W 3 
    else:
        img_flo = flo
    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()
    
    cv2.imwrite(f'{save_path}/{img1_id}', img_flo[:, :, ::-1])  # BGR format for OpenCV
    # breakpoint()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            
            img1_id = imfile1.split('/')[-1]
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            
            
            # ## PHO_IMPLEMENTATION - warp image1 to image2 using flow_up 
            # _, _, H, W = image1.shape 
            # yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W))
            # grid = torch.stack([xx,yy], dim=0).unsqueeze(0).float().cuda() # 1 2 H W
            # grid = grid + flow_up
            # # normalize grid to [-1,1]
            # grid[:,0,:,:] = 2.0 * grid[:,0,:,:] / (W-1) - 1.0 
            # grid[:,1,:,:] = 2.0 * grid[:,1,:,:] / (H-1) - 1.0 
            # # warp image using grid sample 
            # warped_img = F.grid_sample(image1, grid.permute(0,2,3,1), mode='bilinear', align_corners=True)
            # # save warped image 
            # save_image(warped_img, 'img_warped.jpg', normalize=True)


            # # Warp image1 to match image2 using flow_up
            # image1_warped = warp_image(image1, flow_up)
            # warped_img = image1_warped[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            # cv2.imwrite('img_warped2.png', warped_img[:, :, ::-1])
            
            viz(image1, flow_up, img1_id, args.save_path, args.concat_flow)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    
    # pho 
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--concat_flow', action='store_true')
    args = parser.parse_args()

    demo(args)
