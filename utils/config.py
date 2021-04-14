__config__ = {
    'data_provider': 'data.mpii.data_provider',
    'gen_network': 'model.generator.Generator',
    'pose_disc_network': 'model.discriminator.Pose_Discriminator',
    'conf_disc_network': 'model.discriminator.Confidence_Discriminator',
    'inference': {
        'nstack': 2,
        'inp_dim': 256,
        'out_dim': 16,
        'num_parts': 16,
        'increase': 0,
        'keys': ['imgs'],
        'num_eval': 2958, ## number of val examples used. entire set is 2958
        'train_num_eval': 300, ## number of train examples tested at test time
    },

    'train': {
        'batchsize': 8,
        'input_res': 256,
        'output_res': 64,
        'train_iters': 1000,
        'valid_iters': 10,
        'learning_rate': 1e-3,
        'max_num_people' : 1,
        'loss': [
            ['loss', 1],
        ],
        'decay_iters': 100000,
        'decay_lr': 2e-4,
        'num_workers': 2,
        'use_data_loader': True,
        'alpha': 1/180,
        'beta': 1/200,
        'dlta': 0.0005,
    },
}