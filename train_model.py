import os
import tqdm
import shutil
from os.path import dirname

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True

import torch
import numpy as np
import importlib
import argparse
from torch import nn
from torch.nn import DataParallel
from utils.misc import make_input, make_output, importNet
from utils.checkpoints import save_checkpoint, save, reload
import utils.losses as loss
from datetime import datetime
from pytz import timezone
from utils.model_summary import summary as get_summary
#from multiprocessing import set_start_method
#set_start_method('spawn')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#os.chdir('/home/suryam/adversarial-posenet/')

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue_exp', type=str, help='continue exp')
    parser.add_argument('-e', '--exp', type=str, default='pose', help='experiments name')
    parser.add_argument('-m', '--max_iters', type=int, default=200, help='max number of iterations (thousands)')
    args = parser.parse_args()
    return args

class Forward_Generator(nn.Module):
    """
    The wrapper module that will behave differntly for training or testing
    inference_keys specify the inputs for inference
    """
    def __init__(self, gen_net, inference_keys, calc_gen_loss=None):
        super(Forward_Generator, self).__init__()
        self.generator = gen_net
        self.keys = inference_keys
        self.calc_gen_loss = calc_gen_loss

    def forward(self, imgs, **inputs):
        inps = {}
        labels = {}

        for i in inputs:
            if i in self.keys:
                inps[i] = inputs[i]
            else:
                labels[i] = inputs[i]

        if not self.training:
            combined_hm_preds, processed_img = self.generator(imgs, **inps)
            combined_hm_preds = torch.stack(combined_hm_preds, 1)
            if type(combined_hm_preds) != list and type(combined_hm_preds) != tuple:
                combined_hm_preds = [combined_hm_preds]
            return combined_hm_preds
        else:
            combined_hm_preds, processed_img = self.generator(imgs, **inps)
            # generator output will be input for the two discriminators
            discriminator_input = [processed_img] + combined_hm_preds

            combined_hm_preds = torch.stack(combined_hm_preds, 1)
            if type(combined_hm_preds) != list and type(combined_hm_preds) != tuple:
                combined_hm_preds = [combined_hm_preds]

            # calculates gen loss
            gen_loss = self.calc_gen_loss(**labels, combined_hm_preds=combined_hm_preds)

            return list([discriminator_input]) + list(combined_hm_preds) + list([gen_loss])


class Forward_PoseDiscriminator(nn.Module):
    """
    The wrapper module that will behave differetly for training or testing
    inference_keys specify the inputs for inference
    """
    def __init__(self, pdisc_net, inference_keys, calc_pdisc_loss=None):
        super(Forward_PoseDiscriminator, self).__init__()
        self.pose_disc = pdisc_net
        self.keys = inference_keys
        self.calc_pdisc_loss = calc_pdisc_loss

    def forward(self, tag, gen_out, dlta, **inputs):
        inps = {}
        labels = {}

        for i in inputs:
            if i in self.keys:
                inps[i] = inputs[i]
            else:
                labels[i] = inputs[i]

        if tag == 'real':
            p_real = self.pose_disc(tag=tag, pp_img=gen_out[0], **labels)
            p_disc_loss = self.calc_pdisc_loss(tag, p_real, dlta, **labels)
        else:
            p_fake = self.pose_disc(tag=tag, pp_img=gen_out[0], heatmaps=gen_out[1:])
            p_disc_loss = self.calc_pdisc_loss(tag, p_fake, dlta, **labels)

        return p_disc_loss

def make_network(configs):
    train_config = configs['train']
    config = configs['inference']

    def calc_gen_loss(*args, **kwargs):
        return multitaskGen.calc_loss(*args, **kwargs)

    def calc_pdisc_loss(*args, **kwargs):
        return pose_discriminator.calc_loss(*args, **kwargs)

    ## creating adversarial posenet
    # Multi Task Generator
    multitaskGen = importNet(configs['gen_network'])(**config)
    forward_genNet = DataParallel(multitaskGen).cuda()
    # Pose Discriminator
    pose_discriminator = importNet(configs['pose_disc_network'])(**config)
    forward_pDisc = DataParallel(pose_discriminator).cuda()

    config['gen_net'] = Forward_Generator(forward_genNet, configs['inference']['keys'], calc_gen_loss)
    config['pdisc_net'] = Forward_PoseDiscriminator(forward_pDisc, configs['inference']['keys'], calc_pdisc_loss)

    ## optimizer, experiment setup
    train_config['gen_optimizer'] = torch.optim.Adam(filter(lambda p: p.requires_grad, config['gen_net'].parameters()),
                                                 train_config['learning_rate'])
    train_config['pdisc_optimizer'] = torch.optim.Adam(filter(lambda p: p.requires_grad, config['pdisc_net'].parameters()),
                                                 train_config['learning_rate'])

    exp_path = os.path.join('exp', configs['opt'].exp)
    if configs['opt'].exp == 'pose' and configs['opt'].continue_exp is not None:
        exp_path = os.path.join('exp', configs['opt'].continue_exp)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    logger = open(os.path.join(exp_path, 'log'), 'a+')

    def make_train(batch_id, config, phase, **inputs):
        for i in inputs:
            try:
                inputs[i] = make_input(inputs[i])
            except:
                pass  # for last input, which is a string (id_)

        gen_net = config['inference']['gen_net']
        config['batch_id'] = batch_id
        #print(gen_net)
        gen_net = gen_net.train()

        pose_disc = config['inference']['pdisc_net']
        pose_disc = pose_disc.train()

        if phase != 'inference':
            ## Forward Multi-task Generator
            # generator output: list([discriminator_input]) + list(combined_hm_preds) + list([gen_loss])
            gen_out = gen_net(inputs['imgs'], **{i: inputs[i] for i in inputs if i != 'imgs'})

            # slicing gen_out
            disc_ip = gen_out[0]
            gen_losses = gen_out[1:]
            num_loss = len(config['train']['loss']) # num_loss=1

            ## initializing losses
            gen_loss = 0
            pdisc_real_loss = 0
            pdisc_fake_loss = 0
            pdisc_loss = 0

            if phase == 'train':
                ## Pose Discriminator
                # '''pose discriminator output: list([p_real_loss/p_fake_loss])'''
                ## Train Pose Discriminator using fake and real heatmaps
                pdisc_optimizer = train_config['pdisc_optimizer']
                pdisc_optimizer.zero_grad()
                for tag in ['fake', 'real']:
                    pdisc_losses = pose_disc(tag, disc_ip, config['train']['dlta'], **{i: inputs[i] for i in inputs if i != 'imgs'})
                    if tag == 'real':
                        # Phase 1
                        pdisc_real_loss = pdisc_real_loss + torch.mean(pdisc_losses)
                        # optimize P net by maximizing p_real_loss
                        pdisc_real_loss.backward(retain_graph=True)
                    else:
                        # Phase 2
                        pdisc_fake_loss = pdisc_fake_loss + torch.mean(pdisc_losses)
                        # optimize P net by maximizing p_fake_loss
                        pdisc_fake_loss.backward(retain_graph=True)
                    # update weights
                    pdisc_optimizer.step()

                # evaluate gen loss
                gen_loss = gen_loss + torch.mean(gen_losses[-1])

                # update gen_loss
                pdisc_loss = pdisc_real_loss.item() + pdisc_fake_loss.item()


                gen_loss = gen_loss + config['train']['beta'] * (- pdisc_loss)

                ## Train Generator using the updated loss
                gen_optimizer = train_config['gen_optimizer']
                gen_optimizer.zero_grad()
                # optimize generator by updated gen_loss
                gen_loss.backward()
                gen_optimizer.step()

            # printing loss
            toprint = '\n{}: '.format(batch_id)
            genloss = make_output(gen_losses[-1])
            genloss = genloss.mean()
            if pdisc_loss !=0:
                pdiscloss = pdisc_loss
                combined_loss = genloss + config['train']['beta'] * (- pdiscloss)
                toprint += 'gen_loss {} pdisc_loss {} combined_loss {}'.format(format(genloss.mean(), '.8f'), format(-pdiscloss, '.8f'), format(combined_loss.mean(), '.8f'))
                logger.write(toprint)
                logger.flush()

            if batch_id == config['train']['decay_iters']:
                ## decrease the learning rate after decay # iterations
                for param_group in gen_optimizer.param_groups:
                    param_group['lr'] = config['train']['decay_lr']

            return None

        else:
            out = {}
            gen_net = gen_net.eval()
            gen_out = gen_net(**inputs)
            gen_out = gen_out[0]
            if type(gen_out) != list and type(gen_out) != tuple:
                gen_out = [gen_out]
            out['preds'] = [make_output(i) for i in gen_out]
            return out

    return make_train


def train(data_loader, train_func, config):
    while True:
        for phase in ['train', 'valid']:
            num_step = config['train']['{}_iters'.format(phase)]
            data_gen = data_loader(phase)
            print('start', phase, config['opt'].exp)

            show_range = range(num_step)
            show_range = tqdm.tqdm(show_range, total=num_step, ascii=True)
            batch_id = num_step * config['train']['epoch']
            if batch_id > config['opt'].max_iters * 1000:
                return
            for i in show_range:
                data = next(data_gen)
                outs = train_func(batch_id+i, config, phase, **data)
        config['train']['epoch'] += 1
        save(config)


def init():
    """
    utils.__config__ contains the variables and hyperparameters that control the training and testing.
    """
    ## setting hyper parameters
    opt = parse_command_line() # limited control through command line
    task = importlib.import_module('utils.config')
    exp_path = os.path.join('exp', opt.exp)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    config = task.__config__
    try: os.makedirs(exp_path)
    except FileExistsError: pass

    config['opt'] = opt # adding parsed commands to the configuration dict
    config['data_provider'] = importlib.import_module(config['data_provider'])

    ## train model
    train_func = make_network(config)

    # save and reload model
    reload(config)

    return train_func, config


if __name__ == '__main__':

    ## get training function and config
    train_func, config = init()

    ## creating data_loader(), object to load data
    data_loader = config['data_provider'].init(config)

    train(data_loader, train_func, config)
    print(datetime.now(timezone('EST')))