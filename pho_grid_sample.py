import torch 
import torch.nn.functional as F 


# 1. default self sampling 

H, W = 6, 6

t1 = torch.arange(1*3*H*W).view(1,3,H,W).float() 

yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W))
grid = torch.stack([xx, yy], dim=0).unsqueeze(0).float()  # (1, 2, H, W)

grid[:,0,:,:] = 2.0 * grid[:,0,:,:] / max(W - 1, 1) - 1.0 
grid[:,1,:,:] = 2.0 * grid[:,1,:,:] / max(H - 1, 1) - 1.0

t2 = F.grid_sample(t1, grid.permute(0,2,3,1), mode='bilinear', align_corners=True)



# 2. 

H, W = 6, 6

t1 = torch.arange(1*3*H*W).view(1,3,H,W).float() 

H, W = H/2, W/2

yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W))
grid = torch.stack([xx, yy], dim=0).unsqueeze(0).float()  # (1, 2, H, W)



print(grid)

grid[:,0,:,:] = 2.0 * grid[:,0,:,:] / (W - 1) - 1.0 
grid[:,1,:,:] = 2.0 * grid[:,1,:,:] / (H - 1) - 1.0

t2 = F.grid_sample(t1, grid.permute(0,2,3,1), mode='bilinear', align_corners=True)


breakpoint()
