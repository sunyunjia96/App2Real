import torch
from elea import ImagerLoader as ELEA
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import os
import argparse
from models import MULTModel
from transformers import BertModel
from utils import *

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
hyp_params.output_dim = 5
hyp_params.lonly = True
hyp_params.vonly = True
hyp_params.aonly = True

# L1 distance list
def L1_dist(output,label):
    dist = torch.abs(output-label).tolist()
    return dist

# input the whole prediction and label
def mean_accuracy_normalized(output,label):
    dist = torch.abs(output-label)
    normalize = torch.abs(label-torch.mean(label,dim=0))
    return 1-torch.mean(dist,dim=0)/torch.sum(normalize,dim=0)

def mean_accuracy(output,label):
    dist = torch.abs(output-label)
    return 1-torch.mean(dist,dim=0)

def R2(output,label):
    dist = torch.square(output-label)
    normalize = torch.square(label-torch.mean(label,dim=0))
    return 1-torch.sum(dist,dim=0)/torch.sum(normalize,dim=0)

def get_ckpt(path):
    for file in os.listdir(path):
        if file[-8:]=='.pth.tar':
            return os.path.join(path,file)

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

# for average_mode in ['avg','clean','weighted','weighted_clean']:
for average_mode in ['clean']:

    print('**********'+average_mode+'**********')
    
    for rnd in [str(i) for i in range(1,6)]:
        print('**********'+rnd+'**********')

        # network
        model = MULTModel(hyp_params)
        # 5 outputs
        if average_mode =='avg':
            # model.load_state_dict(torch.load('checkpoint/new_label/new_split/plain/personality_average/18.pth.tar'))
            model.load_state_dict(torch.load(get_ckpt('personality_avg_wloss')))
        elif average_mode == 'clean':
            # model.load_state_dict(torch.load('checkpoint/new_label/new_split/plain/personality_clean_avg/3.pth.tar'))
            model.load_state_dict(torch.load(get_ckpt('checkpoint/repeat_exp/plain/clean/personality_clean_'+rnd)))
        elif average_mode=='weighted':
            # model.load_state_dict(torch.load('checkpoint/new_label/new_split/plain/personality_weighted_avg/7.pth.tar'))
            # model.load_state_dict(torch.load('checkpoint/new_label/new_split/wloss/personality_weighted_wloss/15.pth.tar'))
            model.load_state_dict(torch.load(get_ckpt('checkpoint/repeat_exp/personality_weighted_wloss_'+rnd)))
        elif average_mode=='weighted_clean':
            # model.load_state_dict(torch.load('checkpoint/new_label/new_split/plain/personality_weighted_avg_start_from_clean/23.pth.tar'))
            # model.load_state_dict(torch.load('checkpoint/new_label/new_split/wloss/personality_weighted_clean_wloss/15.pth.tar'))
            model.load_state_dict(torch.load('weighted_clean_pred_bias_0.05/23.pth.tar'))
        model = model.cuda()
        bert = BertModel.from_pretrained('/sunyunjia/tools/transformers_models/bert-base-uncased')
        bert = bert.cuda()
        bert.eval()

        with torch.no_grad():
            # validation
            model.eval()
            # dist = []
            pred = torch.tensor([]).cuda()
            gt = torch.tensor([]).cuda()
            for sample in test_loader:
                image_seq = sample['video'][0]
                head_seq = sample['head'][0]
                audio = sample['audio'][0]
                text = sample['text'][0]
                att_mask = sample['att_mask'][0]
                label = sample['label']

                image_seq = image_seq.cuda()
                head_seq = head_seq.cuda()
                audio = audio.cuda()
                text = text.cuda()
                att_mask = att_mask.cuda()
                text = bert(text, attention_mask=att_mask).last_hidden_state
                label = label.cuda()

                output,_ = model(text,audio,image_seq,head_seq)
                # output = output[:,:5]

                # get new label
                if average_mode=='avg':
                    output = output.mean(dim=0,keepdim=True)
                elif average_mode=='clean':
                    output = torch.tensor(get_clean_avg(output)).unsqueeze(0).cuda()
                elif average_mode=='weighted':
                    output = torch.tensor(get_weighted_avg(output)).unsqueeze(0).cuda()
                elif average_mode=='weighted_clean':
                    output = torch.tensor(get_weighted_avg(output,'clean')).unsqueeze(0).cuda()

                pred = torch.cat((pred,output),dim=0)
                gt = torch.cat((gt,label),dim=0)

            print(f'MA: {mean_accuracy(pred,gt).tolist()}\t')
            print(f'R2: {R2(pred,gt).tolist()}\t')
