import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler, coords_grid

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, kernel_size=2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)     # 1 h w 2 
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):    # 4
            corr = self.corr_pyramid[i]                                     # 7040 1 55 128 
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)            # 9 9 2

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i      # 7040 1 1 2
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)                      # 1 9 9 2
            coords_lvl = centroid_lvl + delta_lvl                           # 7040 9 9 2 

            '''
                현재 두 feature map src, trg (H,W,C) 라고 하고,
                이들의 correlation (cost volume)은 (H,W,H,W) 이다. 
                즉, src의 각 모든 픽셀별로 trg의 모든 픽셀과의 similarity가 구해짐.
                즉 corr = (HW, HW) 꼴, 
                여기서 우리가 하고 싶은 것은, src의 각 픽셀별로 HW 만큼 correspondence가 너무 비싸기에, 픽셀별로 근방 window안에서만 correspondence를 구하고 싶다.
                window크기를 9x9로 한다면, src의 HW개의 각 픽셀은 Trg의 9x9개의 픽셀과만 correspondence로 줄일 수 있따.
                즉 corr (HW,HW)에서 -> corr (HW, 81)로 줄일 수 있는 것. 
                여기서 dx, dy는 해당 픽셀 위치를 centroid로 하여, 주변 9x9 window의 grid를 생성하는 역할을 하고,
                bilinear_sample()는 다른게 아니고 그냥 F.grid_sample()과 같은 역할을 하는데, 
                corr (HW, 1, H, W)) 에서 어떤 grid를 넣어주면, 그 grid에 해당하는 위치들의 픽셀들을 가져오게 되는 것. 
            '''
            corr = bilinear_sampler(corr, coords_lvl)                       # 7040 1 9 9 
            corr = corr.view(batch, h1, w1, -1)                             # 1 55 128 81
            out_pyramid.append(corr)

        '''
            out_pyramid = [corr_lvl0, corr_lvl1, corr_lvl2, corr_lvl3]
            corr_lvl0.shape = (1, 55, 128, 81)
            corr_lvl1.shape = (1, 55, 128, 81)
            corr_lvl2.shape = (1, 55, 128, 81)
            corr_lvl3.shape = (1, 55, 128, 81)
        '''
        out = torch.cat(out_pyramid, dim=-1)    # concat the trg correlation informations (1 55 128 81+81+81+81)
        return out.permute(0, 3, 1, 2).contiguous().float() # 1 c1+c2+c3+c4 55 128

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        corr = corr  / torch.sqrt(torch.tensor(dim).float())
        return corr


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())
