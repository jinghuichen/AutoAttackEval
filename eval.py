# CIFAR10 checkpoint available at https://drive.google.com/file/d/1lvMa2rbMrIVkAqsyrs_YXLBhewZBfdkP/view?usp=sharing
# CIFAR100 chekcpoint available at https://drive.google.com/file/d/1xNhK4w5ZuUSfbD_WR4xFKTprojaVux1A/view?usp=sharing


import os
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

import sys
sys.path.insert(0,'..')

from wideresnet import *
from func import DataAugmentModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=float, default=8./255.)
    parser.add_argument('--model', type=str, default='/model_test.pt')
    parser.add_argument('--n_ex', type=int, default=10000)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--log_path', type=str, default='./log_file.txt')
    parser.add_argument('--version', type=str, default='standard')
    
    args = parser.parse_args()

    model = WideResNet()
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    im_mean = torch.tensor(cifar10_mean).cuda().view(1, 3, 1, 1)
    im_std = torch.tensor(cifar10_std).cuda().view(1, 3, 1, 1)
    model = DataAugmentModel(model, im_mean=im_mean, im_std=im_std)

    # model = WideResNet(num_classes=100)
    # cifar100_mean = (0.5071, 0.4867, 0.4408)
    # cifar100_std = (0.2675, 0.2565, 0.2761)
    # im_mean = torch.tensor(cifar100_mean).cuda().view(1, 3, 1, 1)
    # im_std = torch.tensor(cifar100_std).cuda().view(1, 3, 1, 1)
    # model = DataAugmentModel(model, im_mean=im_mean, im_std=im_std)

    ckpt = torch.load(args.model)
    model.load_state_dict(ckpt)

    model.cuda()
    model.eval()

    # load data
    transform_list = [transforms.ToTensor()]
    transform_chain = transforms.Compose(transform_list)
    item = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_chain, download=True)
    test_loader = data.DataLoader(item, batch_size=1000, shuffle=False, num_workers=0)
    
    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # load attack    
    from autoattack import AutoAttack
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path,
        version=args.version)
    
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    
    # example of custom version
    if args.version == 'custom':
        adversary.attacks_to_run = ['apgd-ce', 'fab']
        adversary.apgd.n_restarts = 2
        adversary.fab.n_restarts = 2
    
    # run attack and save images
    with torch.no_grad():
        if not args.individual:
            adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
                bs=args.batch_size)
            
            torch.save({'adv_complete': adv_complete}, '{}/{}_{}_1_{}_eps_{:.5f}.pth'.format(
                args.save_dir, 'aa', args.version, adv_complete.shape[0], args.epsilon))

        else:
            # individual version, each attack is run on all test points
            adv_complete = adversary.run_standard_evaluation_individual(x_test[:args.n_ex],
                y_test[:args.n_ex], bs=args.batch_size)
            
            torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}_plus_{}_cheap_{}.pth'.format(
                args.save_dir, 'aa', args.version, args.n_ex, args.epsilon))
                
