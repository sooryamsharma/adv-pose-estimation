import importlib
import torch
import os
import numpy as np
import cv2
from skimage.transform import resize
import matplotlib.pyplot as plt
import data.mpii.mpii_data_handler as data_handler
import data.mpii.data_provider as data_provider
from utils.transparent_imshow import transp_imshow
from utils.checkpoints import save_checkpoint, save, reload
from scipy.misc import imread

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def get_maxprob_coordinates(pred_hm_mat):
    h, w = pred_hm_mat.shape
    pred_hm_mat = pred_hm_mat.reshape(-1)
    idx = np.argmax(pred_hm_mat)
    return idx % w, idx // w

def generate_hm(config, keypoints):
    get_hm = data_provider.GenerateHeatmap(config['train']['output_res'], config['inference']['num_parts'])
    return get_hm(keypoints)

def get_preds(config, test_img):

    # np img to tensor
    test_img = torch.FloatTensor(test_img)
    test_img = test_img.view(1,256,256,3)

    # load check point
    checkpoint_fpath = 'exp/pose/checkpoint.pt'
    checkpoint = torch.load(checkpoint_fpath)

    # save and reload model
    reload(config)
    gen_net = config['inference']['gen_net']
    gen_net.load_state_dict(checkpoint['state_dict'])
    gen_net = gen_net.eval()
    gen_out = gen_net(test_img)
    gen_out = gen_out[0]

    # processing o/p
    preds = torch.squeeze(gen_out)
    preds = torch.squeeze(preds[1])
    preds = torch.Tensor.cpu(preds).detach().numpy()

    return preds

def join_kp(kp):
    kp = (kp*256)/64
    plt.plot([kp[0][0][0], kp[0][1][0]], [kp[0][0][1], kp[0][1][1]], 'g-') # left ankle - > left knee
    plt.plot([kp[0][1][0], kp[0][2][0]], [kp[0][1][1], kp[0][2][1]], 'g-') # left knee - > left hip
    plt.plot([kp[0][2][0], kp[0][6][0]], [kp[0][2][1], kp[0][6][1]], 'g-')  # left hip - > pelvis
    plt.plot([kp[0][6][0], kp[0][3][0]], [kp[0][6][1], kp[0][3][1]], 'b-')  # pelvis - > right hip
    plt.plot([kp[0][3][0], kp[0][4][0]], [kp[0][3][1], kp[0][4][1]], 'b-')  # right hip - > right knee
    plt.plot([kp[0][4][0], kp[0][5][0]], [kp[0][4][1], kp[0][5][1]], 'b-')  # right knee - > right ankle
    plt.plot([kp[0][6][0], kp[0][7][0]], [kp[0][6][1], kp[0][7][1]], 'm-')  # pelvis - > thorax
    plt.plot([kp[0][7][0], kp[0][8][0]], [kp[0][7][1], kp[0][8][1]], 'm-')  # thorax - > neck
    plt.plot([kp[0][8][0], kp[0][9][0]], [kp[0][8][1], kp[0][9][1]], 'r-')  # neck - > head
    plt.plot([kp[0][10][0], kp[0][11][0]], [kp[0][10][1], kp[0][11][1]], 'c-')  # left wrist - left elbow
    plt.plot([kp[0][11][0], kp[0][12][0]], [kp[0][11][1], kp[0][12][1]], 'c-')  # left elbow - left shoulder
    plt.plot([kp[0][12][0], kp[0][8][0]], [kp[0][12][1], kp[0][8][1]], 'c-')  # left shoulder - neck
    plt.plot([kp[0][8][0], kp[0][13][0]], [kp[0][8][1], kp[0][13][1]], 'y-')  # neck - right shoulder
    plt.plot([kp[0][13][0], kp[0][14][0]], [kp[0][13][1], kp[0][14][1]], 'y-')  # right shoulder - right elbow
    plt.plot([kp[0][14][0], kp[0][15][0]], [kp[0][14][1], kp[0][15][1]], 'y-') # right elbow - right wrist

def main():
    from train_model import init
    func, config = init()

    # creating training data
    data_handler.init()
    train, valid = data_handler.setup_val_split()
    data = [train, valid]

    # get configurations
    config = importlib.import_module('utils.config').__config__
    data_provider.init(config)

    # get image idx from annot file
    img_id = "099825179"
    img_path = img_id+'.jpg'
    idx = data_handler.get_imgID(img_path)

    # get image and heat map
    ds = data_provider.Dataset(config=config, ds=data_handler, index=data)
    img, gt_hmap, gt_kp = ds.loadOrigImage(1729)  # input image index
    plt.imshow(img)
    join_kp(gt_kp)

    for i in range(gt_hmap.shape[0]):
        hm = gt_hmap[i, :, :]
        hm = resize(hm, (256, 256), anti_aliasing=True)
        transp_imshow(hm, cmap='hsv')  # custom function
    plt.show()

    preds = get_preds(config, img)
    pred_keypoints = np.ones((1, 16, 3))

    for i in range(preds.shape[0]):
        mat = preds[i, :, :]
        x, y = get_maxprob_coordinates(mat)
        pred_keypoints[0][i][1] = y
        pred_keypoints[0][i][0] = x

    pred_hmap = generate_hm(config, pred_keypoints)

    plt.imshow(img)
    join_kp(pred_keypoints)

    for i in range(pred_hmap.shape[0]):
        hm = pred_hmap[i, :, :]
        hm = resize(hm, (256, 256), anti_aliasing=True)
        transp_imshow(hm, cmap='hsv')  # custom function
    plt.show()

if __name__ == '__main__':
    main()