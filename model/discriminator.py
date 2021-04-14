import torch
from torch import nn
from model.layers import Conv, Hourglass, Pool, Residual
from utils.losses import PoseDiscriminatorLoss

class Pose_Discriminator(nn.Module):
    def __init__(self, nstack, inp_dim, out_dim, bn=False, **kwargs):
        super(Pose_Discriminator, self).__init__()
        self.nstack = 1 # discriminator is a single stack hourglass network
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

        self.get_pDiscLoss = PoseDiscriminatorLoss()

    def forward(self, tag, pp_img, heatmaps):
        disc_out = []
        # merging pose and occlusion heatmaps
        x = pp_img
        if tag =='fake':
            len = heatmaps.__len__()
            for i in range(len):
                x = x + self.merge_heatmaps[i](heatmaps[i])
        else:
            for i in range(1):
                x = x + self.merge_heatmaps[i](heatmaps)
        for i in range(self.nstack):
            hg_out = self.hourglass[i](x)
            feature = self.features[i](hg_out)
            disc_out = self.heatmaps[i](feature)

        return disc_out

    def calc_loss(self, tag, pred, dlta, heatmaps):

        pdisc_loss = self.get_pDiscLoss(tag, pred, dlta, heatmaps)
        '''
        pdisc_loss = []
        for i in range(self.nstack):
            pdisc_loss.append(self.get_pDiscLoss(p_real, p_fake))
        pdisc_loss = torch.stack(pdisc_loss, dim=1)
        '''
        return pdisc_loss