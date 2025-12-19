import torch
from dataloaders import FirstImpressionV2_no_id_overlap as FirstImpressionV2
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import os
import argparse
from models import MULTModel
from transformers import BertModel
import torch.optim as optim
from loss_functions import VarianceLoss

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

# L1 distance list
def metric(output,label):
    dist = torch.abs(output-label)
    dist = dist.sum(dim=1)
    return dist.tolist()

# dataset
image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
batch_size = 8
num_workers = 4

transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            #image_normalize,
            ])

transform_aug = transforms.Compose([
            transforms.ColorJitter(brightness=0.6, contrast=0.7, saturation=0.5, hue=0.1),
            transforms.RandomRotation(30, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop((224,224),(0.8,1.0)),
            # transforms.Resize((224,224)),
            transforms.ToTensor(),
            #image_normalize,
            ])

average = 'clean'
print('*************'+average+'*************')

# train
train_frame_root = '/sunyunjia/data/personality/FirstImpressionV2/frames/'
train_label_path = '/sunyunjia/data/personality/FirstImpressionV2/rearranged_labels/'
train_audio_root = '/sunyunjia/data/personality/FirstImpressionV2/audios/'
train_text_path = '/sunyunjia/data/personality/FirstImpressionV2/rearranged_labels/text.json'
train_head_box_root = '/sunyunjia/data/personality/FirstImpressionV2/landmarks/'
train_loader = torch.utils.data.DataLoader(
        FirstImpressionV2(
        train_frame_root, train_audio_root,
        train_label_path, train_text_path, ['train'],
        average_mode = average,
        transform = transform,
        # transform_aug = transform_aug,
        head_box_root=train_head_box_root,
        ),
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True)

# validation
val_frame_root = '/sunyunjia/data/personality/FirstImpressionV2/frames/'
val_label_path = '/sunyunjia/data/personality/FirstImpressionV2/rearranged_labels/'
val_audio_root = '/sunyunjia/data/personality/FirstImpressionV2/audios/'
val_text_path = '/sunyunjia/data/personality/FirstImpressionV2/rearranged_labels/text.json'
val_head_box_root = '/sunyunjia/data/personality/FirstImpressionV2/landmarks/'
val_loader = torch.utils.data.DataLoader(
        FirstImpressionV2(
        val_frame_root, val_audio_root,
        val_label_path, val_text_path, ['val'],
        average_mode = average,
        transform = transform,
        # transform_aug = transform_aug,
        head_box_root=val_head_box_root,
        ),
		batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True)

set_wloss =True
# train parameters
bias_range = 0.1
if set_wloss:
    save_folder = f'{average}_pred_bias_wloss_detach_{bias_range}_5'
else:
    save_folder = f'{average}_pred_bias_{bias_range}_5'
os.makedirs(save_folder,exist_ok=True)

# network
model = MULTModel(hyp_params)
# 5 outputs
model = model.cuda()
bert = BertModel.from_pretrained('/sunyunjia/tools/transformers_models/bert-base-uncased')
bert.requires_grad_(False)
bert = bert.cuda()
bert.eval()

epochs = 30
lr = 1e-4
if set_wloss:
    criterion = nn.L1Loss(reduction='none')
else:
    criterion = nn.L1Loss()
var_loss = VarianceLoss(target_var=0.01,normalize=False)
optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1, last_epoch=-1)


for epoch in range(epochs):
    # train
    model.train()
    running_loss = 0.
    dist = []
    for i,sample in enumerate(train_loader):
        image_seq = sample['video']
        head_seq = sample['head']
        audio = sample['audio']
        text = sample['text']
        att_mask = sample['att_mask']
        label = sample['label']
        org_label = sample['org_label']

        image_seq = image_seq.cuda()
        head_seq = head_seq.cuda()
        audio = audio.cuda()
        text = text.cuda()
        att_mask = att_mask.cuda()
        text = bert(text, attention_mask=att_mask).last_hidden_state
        label = label.cuda()
        org_label = org_label.cuda()

        output,_ = model(text,audio,image_seq,head_seq)
        pred_bias = output[:,5:]
        output = output[:,:5]
        label = label+torch.sigmoid(pred_bias)*bias_range*2-bias_range
        loss = criterion(output,label)
        if set_wloss:
            loss_w = 0.1/torch.abs(org_label-label)
            loss_w = loss_w.clamp(max=10)
            loss_w = loss_w.detach().clone()
            loss = torch.mean(loss*loss_w)
        loss += var_loss(label)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        dist += metric(output,label)
        if i % 100 == 99:
            print(f'Train: {epoch}\t'
                    f'Iteration: {i+1}\t'
                    f'L1 Dist: {np.mean(dist)}\t'
                    f'Loss: {running_loss/100}')
            running_loss = 0.
            dist = []
    
    scheduler.step()
    with torch.no_grad():
        # validation
        model.eval()
        torch.save(model.state_dict(),os.path.join(save_folder,f'{epoch}.pth.tar'))
        dist = []
        dist_bias = []
        for sample in val_loader:
            image_seq = sample['video']
            head_seq = sample['head']
            audio = sample['audio']
            text = sample['text']
            att_mask = sample['att_mask']
            label = sample['label']
            
            image_seq = image_seq.cuda()
            head_seq = head_seq.cuda()
            audio = audio.cuda()
            text = text.cuda()
            att_mask = att_mask.cuda()
            text = bert(text, attention_mask=att_mask).last_hidden_state
            label = label.cuda()

            output,_ = model(text,audio,image_seq,head_seq)
            pred_bias = output[:,5:]
            output = output[:,:5]
            dist += metric(output,label)
            label = label+torch.sigmoid(pred_bias)*bias_range*2-bias_range
            dist_bias += metric(output,label)

        print(f'Validation: {epoch}\t'
                f'L1 Dist: {np.mean(dist)}\t'
                f'L1 Dist Biased: {np.mean(dist_bias)}')
