import torch
import os
import shutil
from os.path import dirname


def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    """
    from pytorch/examples
    """
    basename = dirname(filename)
    if not os.path.exists(basename):
        os.makedirs(basename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pt')


def save(config):
    resume = os.path.join('exp', config['opt'].exp)
    if config['opt'].exp=='pose' and config['opt'].continue_exp is not None:
        resume = os.path.join('exp', config['opt'].continue_exp)
    resume_file = os.path.join(resume, 'checkpoint.pt')

    save_checkpoint({
            'state_dict': config['inference']['gen_net'].state_dict(),
            'gen_optimizer' : config['train']['gen_optimizer'].state_dict(),
            'epoch': config['train']['epoch'],
        }, False, filename=resume_file)
    print('=> save checkpoint')

def reload(config):
    """
    load or initialize model's parameters by config from config['opt'].continue_exp
    config['train']['epoch'] records the epoch num
    config['inference']['gen_net'] is the model
    """
    opt = config['opt']

    if opt.continue_exp:
        resume = os.path.join('exp', opt.continue_exp)
        resume_file = os.path.join(resume, 'checkpoint.pt')
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume_file)

            config['inference']['gen_net'].load_state_dict(checkpoint['state_dict'])
            config['train']['gen_optimizer'].load_state_dict(checkpoint['gen_optimizer'])
            config['train']['epoch'] = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            exit(0)

    if 'epoch' not in config['train']:
        config['train']['epoch'] = 0