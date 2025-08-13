import cv2 
from PIL import Image
import torch 
from torchvision.utils import save_image 
import torchvision.transforms as T 
import torch.nn.functional as F
from core.utils import flow_viz


# Load and preprocess image
img = Image.open('./corgi.png').convert('RGB')
trans = T.Compose([
    T.Resize((224, 448)),   # Resize to (H, W)
    T.ToTensor(),
])
img = trans(img).unsqueeze(0)  # Add batch dimension -> (1, 3, H, W)

_, _, H, W = img.shape

# dummy flow 
flow = torch.rand(1, 2, H, W)
# flow[:, 0] = -100  # x-direction flow (rightward)

vis_flow = flow_viz.flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy())   # H W 3 
cv2.imwrite('./img1.jpg', vis_flow)

breakpoint()
# Convert pixel coordinates + flow into normalized grid for grid_sample
# Normalize flow: [-1, 1] range
yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W))
grid = torch.stack((xx, yy), dim=-1).float()  # (H, W, 2)
grid = grid.unsqueeze(0)  # (1, H, W, 2)
grid = grid + flow.permute(0, 2, 3, 1)  # add flow

# Normalize grid to [-1, 1]
grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / max(W - 1, 1) - 1.0
grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / max(H - 1, 1) - 1.0

breakpoint()
# Warp the image
warped = F.grid_sample(img, grid, mode='bilinear', align_corners=True)  # F.grid_sample expects grid in (N, H, W, 2) format w/ [-1,1] normalization

# Save result
save_image(img, './img2.jpg', normalize=True)
save_image(warped, './img3.jpg', normalize=True)

