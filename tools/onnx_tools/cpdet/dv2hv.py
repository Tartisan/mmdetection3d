import mmcv
from mmcv.runner import load_checkpoint
import torch
import numpy as np
from argparse import ArgumentParser
from collections import OrderedDict


def main():
    parser = ArgumentParser()
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    device = args.device
    checkpoint_dv = torch.load(args.checkpoint, map_location=device)
    checkpoint_hv = {}
    for key in checkpoint_dv.keys():
        if key == 'state_dict':
            checkpoint_hv['state_dict'] = OrderedDict()
            for sd_key in checkpoint_dv['state_dict'].keys():
                if 'pfn' in sd_key:
                    if '0.0' in sd_key:
                        new_key = sd_key.replace('0.0', '0.linear')
                    elif '0.1' in sd_key:
                        new_key = sd_key.replace('0.1', '0.norm')
                else:
                    new_key = sd_key
                checkpoint_hv['state_dict'][new_key] = checkpoint_dv['state_dict'][sd_key]
        else:
            checkpoint_hv[key] = checkpoint_dv[key]

    
    new_checkpoint_file = args.checkpoint.split('.pth')[0] + '_hv.pth'
    with open(new_checkpoint_file, 'wb') as f:
        torch.save(checkpoint_hv, f)
    


if __name__ == '__main__':
    main()
