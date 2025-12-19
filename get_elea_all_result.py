import torch
from dataloaders import ELEA
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import os
import argparse
from models import MULTModel
from transformers import BertModel
from utils import *
import json

parser = argparse.ArgumentParser(description='FirstImpressionV2 personality analysis')
parser.add_argument('-f', default='', type=str)

# Tasks
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')
parser.add_argument('--aligned', action='store_true',
                    help='consider aligned experiment or not (default: False)')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0,
                    help='output layer dropout')

# Architecture
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=5,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')

# Tuning
parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='mult',
                    help='name of the trial (default: "mult")')
args = parser.parse_args()

hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 768, 1690, 1000
hyp_params.orig_d_h = 1000
hyp_params.layers = args.nlevels
hyp_params.use_cuda = True
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.output_dim = 10
hyp_params.lonly = True
hyp_params.vonly = True
hyp_params.aonly = True

# dataset
image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            #image_normalize,
            ])

frame_root = '/sunyunjia/data/personality/elea/elea/frames/original/'
label_path = '/sunyunjia/data/personality/elea/elea/processed_infos/personality.json'
audio_root = '/sunyunjia/data/personality/elea/elea/audio/Groups1-40_wav/'
transcript_path = '/sunyunjia/data/personality/elea/elea/processed_infos/transcript.json'
pos_path = '/sunyunjia/data/personality/elea/elea/processed_infos/pos_info.json'
landmark_root = '/sunyunjia/data/personality/elea/elea/landmarks/'
audio_delay_file = '/sunyunjia/data/personality/elea/elea/video/audiodelayMS.txt'
test_loader = torch.utils.data.DataLoader(
        ELEA(
        frame_root, audio_root,
        transcript_path, label_path,
        pos_path, landmark_root,
        audio_delay_file,
        transform = transform,
        ),
        batch_size = 1,
        shuffle = False,
        num_workers = 0,
        pin_memory = True)

bert = BertModel.from_pretrained('/sunyunjia/tools/transformers_models/bert-base-uncased')
bert = bert.cuda()
bert.eval()

path_list, dir_list, json_list = get_all_checkpoints('checkpoint/repeat_exp/')

for i in range(len(path_list)):
    ckpt_path = os.path.join('checkpoint/repeat_exp/',path_list[i])
    result_path = os.path.join('results/results/',json_list[i])
    if os.path.exists(result_path):
        continue
    dir_path = os.path.join('results/results/',dir_list[i])
    os.makedirs(dir_path,exist_ok=True)

    # network
    model = MULTModel(hyp_params)
    model.load_state_dict(torch.load(ckpt_path))
    model = model.cuda()

    pred = {}

    with torch.no_grad():
        # validation
        model.eval()
        for sample in test_loader:
            image_seq = sample['video'][0]
            head_seq = sample['head'][0]
            audio = sample['audio'][0]
            text = sample['text'][0]
            att_mask = sample['att_mask'][0]
            label = sample['label']
            pid = sample['pid'][0]

            image_seq = image_seq.cuda()
            head_seq = head_seq.cuda()
            audio = audio.cuda()
            text = text.cuda()
            att_mask = att_mask.cuda()
            text = bert(text, attention_mask=att_mask).last_hidden_state
            label = label.cuda()

            output,_ = model(text,audio,image_seq,head_seq)

            pred[pid] = output[:,:5].tolist()

    with open(result_path,'w') as f:
        json.dump(pred,f,indent=4)