from torch import nn
from torch.nn import functional as F
import torch

def CGAN_loss(logits, target):
    """
    logits dim: (batch_size, out_channels, H, W)
    target dim: (batch_size, pose_channels, H, W)
    """
    # Calculate log probabilities
    log_prob = F.log_softmax(logits)

    target = target.long() # converting target to long

    cross_entropy_loss = []

    for channel in range(logits.shape[1]):
        # Define Weight mask; if needed
        #wt_mask = torch.FloatTensor(8,16,64,64).zero_()
        #wt_mask[:,channel,:,:] = torch.ones(64,64)

        # Extracting target channel
        target_i = target[:,channel,:,:]

        # Gather log probabilities with respect to target
        #log_prob_i = log_prob.gather(1, target_i)
        log_prob_i = log_prob[:,channel,:,:]
        log_prob_i = torch.unsqueeze(log_prob_i, 1)
        log_prob_i = torch.cat((log_prob_i, log_prob_i), 1)

        # Loss
        loss = - F.nll_loss(log_prob_i, target_i.long())/2

        # appending loss for each channel
        cross_entropy_loss.append(loss)

    cross_entropy_loss = torch.stack(cross_entropy_loss, 0)

    return cross_entropy_loss


class GeneratorLoss(nn.Module):
    """
    Calculates loss for generated Heatmaps.
    """
    def __init__(self):
        super(GeneratorLoss, self).__init__()

    def forward(self, pred, gt):
        l = ((pred - gt)**2)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l ## l of dim bsize


class PoseDiscriminatorLoss(nn.Module):
    """
    Calculates loss for Pose Discriminator.
    """
    def __init__(self):
        super(PoseDiscriminatorLoss, self).__init__()

    def forward(self, tag, pred, dlta, gt):

        ## Negative cross entropy loss, to maximize the discriminator loss
        ## L_D = log(D(y)) + log(1 - |D(G(x)) - pfake|)
        ##---------1st term---------2nd term------------------------------
        l = []

        # calculating 1st term of Lp
        if tag == 'real':
            # logits = pred
            # l.append(CGAN_loss(logits, gt))

            pred[pred<1] = 0
            pred[pred>=1] = 1
            l.append(-F.binary_cross_entropy(pred, gt))

        # calculating 2nd term of Lp
        else:
            ## calculate l2 distance
            l2_dist = ((pred - gt) ** 2)
            l2_dist = l2_dist.mean(dim=3).mean(dim=2).mean(dim=1)
            pfake = l2_dist.le(dlta)
            pfake = pfake.float()
            tnsr = pred
            for i, pfake_i in enumerate(pfake):
                pfake_i = pfake_i.item()
                if pfake_i == 0:
                    tmp_tnsr = torch.zeros([16,64,64])
                else:
                    tmp_tnsr = torch.ones([16,64,64])
                tnsr[i,:,:,:] = tmp_tnsr
            pfake = tnsr


            # calculating 2nd term of Lp
            # logits = torch.abs(pred - pfake)
            # logits = 1 - logits
            #
            # gt = 1 - gt
            l.append(-F.binary_cross_entropy(1 - torch.abs(pred - pfake), 1 - gt))

        return l[0]
