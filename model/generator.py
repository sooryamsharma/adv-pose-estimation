import torch
from torch import nn
from model.layers import Conv, Hourglass, Pool, Residual
from utils.losses import GeneratorLoss

class Generator(nn.Module):
    def __init__(self, nstack, inp_dim, out_dim, bn=False, **kwargs):
        super(Generator, self).__init__()

        self.nstack = nstack
        self.pre_process = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )

        self.hourglass = nn.ModuleList([
            nn.Sequential(
                Hourglass(hg_depth=4, nFeatures=inp_dim),
            ) for _ in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim=inp_dim, out_dim=inp_dim),
                Conv(inp_dim=inp_dim, out_dim=inp_dim, kernel_size=1, bn=True, relu=True)
            ) for _ in range(nstack)])

        self.heatmaps = nn.ModuleList([
            Conv(inp_dim=inp_dim, out_dim=out_dim, kernel_size=1, bn=False, relu=False) for _ in range(nstack)])

        self.merge_features = nn.ModuleList([
            Conv(inp_dim=inp_dim, out_dim=inp_dim, kernel_size=1, bn=False, relu=False) for _ in range(nstack)])

        self.merge_heatmaps = nn.ModuleList([
            Conv(inp_dim=out_dim, out_dim=inp_dim, kernel_size=1, bn=False, relu=False) for _ in range(nstack)])

        self.get_genLoss = GeneratorLoss()

    def forward(self, imgs):
        x = imgs.permute(0,3,1,2) #x of size 1,3,inpdim,inpdim
        x = self.pre_process(x)
        processed_img = x
        combined_hm_preds = []
        for i in range(self.nstack):
            hg_out = self.hourglass[i](x)
            feature = self.features[i](hg_out)
            hm = self.heatmaps[i](feature)
            combined_hm_preds.append(hm)
            if i < self.nstack - 1:
                x = x + self.merge_heatmaps[i](hm) + self.merge_features[i](feature)
        return combined_hm_preds, processed_img

    def calc_loss(self, combined_hm_preds, heatmaps):
        gen_loss = []
        for i in range(self.nstack):
            gen_loss.append(self.get_genLoss(combined_hm_preds[0][:,i], heatmaps))
        gen_loss = torch.stack(gen_loss, dim=1)
        return gen_loss

"""
x_1_1 = self.pre_process(x)
pose_hm = []
occlusion_hm = []

## stack 1
#---------------------------------------------------------------------------
hg1_out = self.hourglass[0](x_1_1)
features1 = self.features[0](hg1_out)
hm1 = self.heatmaps[0](features1)
pose_hm.append(hm1)

x_1_2 = x_1_1 + self.merge_heatmaps[0](hm1) + self.merge_features[0](features1)

#---------------------------------------------------------------------------
hg2_out = self.hourglass[1](x_1_2)
features2 = self.features[1](hg2_out)
hm2 = self.heatmaps[1](features2)
occlusion_hm.append(hm2)

x_2_1 = x_1_2 + self.merge_heatmaps[1](hm2) + self.merge_features[1](features2)

## stack 2
# ---------------------------------------------------------------------------
hg3_out = self.hourglass[2](x_2_1)
features3 = self.features[2](hg3_out)
hm3 = self.heatmaps[2](features3)
pose_hm.append(hm3)

x_2_2 = x_2_1 + self.merge_heatmaps[2](hm3) + self.merge_features[2](features3)

# ---------------------------------------------------------------------------
hg4_out = self.hourglass[3](x_2_2)
features4 = self.features[3](hg4_out)
hm4 = self.heatmaps[3](features4)
pose_hm.append(hm4)

out = x_2_2 + self.merge_heatmaps[3](hm4) + self.merge_features[3](features4)
"""