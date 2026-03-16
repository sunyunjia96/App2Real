import torch
from PIL import Image
import torch.utils.data as data
import os
import numpy as np
import torchaudio
from transformers import BertTokenizer
import torch.nn.functional as F
import json
import random
import time

# for ablation
# random choose one sample as label
def get_random_one_dim(input_list):
    return random.choice(input_list)

def get_random(input_list):
    randomed = []
    for i in range(5):
        randomed.append(get_random_one_dim(np.array(input_list)[:,i]))
    return randomed

# wipe off the outliers
# type: std - wipe off data points without the mean+_3*std
# TBD: type: IQR - wipe off data points without (Q1 - 1.5 * IQR) or (Q3 + 1.5 * IQR)
def get_clean_avg_one_dim(input_list, type='std'):
    if type=='std':
        std = np.std(input_list, ddof=1)
        mean = np.mean(input_list)
        if std==0:
            return mean
        lower_bound = mean-1.5*std
        upper_bound = mean+1.5*std
    new_list = []
    for item in input_list:
        if lower_bound < item < upper_bound:
            new_list.append(item)
    return np.mean(new_list)

def get_clean_avg(input_list):
    clean_avg = []
    for i in range(5):
        clean_avg.append(get_clean_avg_one_dim(np.array(input_list)[:,i]))
    return clean_avg

# iteration algorithm
# to give data points that are far away from the initial average less weight
def get_weighted_avg_one_dim(input_list, start):
    if start=='avg':
        w_avg = np.mean(input_list)
    elif start=='clean':
        w_avg = get_clean_avg_one_dim(input_list)
    for i in range(30):
        dist = (input_list - w_avg)**2
        # get un-normalized weights
        w = 1/(dist+1e-6)
        normalized_factor = np.sum(w)
        # normalized weights
        w = w/normalized_factor
        w_avg_new = np.sum(input_list*w)
        if np.abs(w_avg_new-w_avg) < 0.001:
            return w_avg_new
        w_avg = w_avg_new
    return w_avg_new

def get_weighted_avg(input_list,start='avg'):
    traits_w_avg = []
    for i in range(5):
        traits_w_avg.append(get_weighted_avg_one_dim(np.array(input_list)[:,i],start))
    return traits_w_avg

# use all_label.json to get all labels with id sorted
# average: whether use averaged label or the original label
# split: ['xxx',...] 
# average mode: 'none' for not average, 'avg' for regular mean
# 'weighted' for weighted average
def make_dataset(label_root, text_path, split, average_mode):
    label_path = os.path.join(label_root,'all_label.json')
    split_path = os.path.join(label_root,'split.json')

    with open(label_path,'r') as f:
        data = json.load(f)

    with open(split_path,'r') as f:
        all_id = json.load(f)
    used_id = []
    for i in split:
        used_id += all_id[i]

    with open(text_path,'r') as f:
        text = json.load(f)

    video_data = []
    for p in used_id:
        if average_mode!='none':
            # get average
            vectors = []
            for video_name in data[p]:
                vectors.append([data[p][video_name]['extraversion'],
                            data[p][video_name]['neuroticism'],
                            data[p][video_name]['agreeableness'],
                            data[p][video_name]['conscientiousness'],
                            data[p][video_name]['openness']])
            if average_mode=='avg':
                id_avg = np.mean(vectors,axis=0)
            elif average_mode in ['weighted','weighted_clean']:
                if len(vectors)>2:
                    if average_mode=='weighted':
                        id_avg = get_weighted_avg(vectors)
                    else:
                        id_avg = get_weighted_avg(vectors,'clean')
                else:
                    id_avg = np.mean(vectors,axis=0)
            elif average_mode=='clean':
                if len(vectors)>2:
                    id_avg = get_clean_avg(vectors)
                else:
                    id_avg = np.mean(vectors,axis=0)
            elif average_mode=='random':
                id_avg = get_random(vectors)
        for video_name in data[p]:
            if average_mode!='none':
                video_data.append([video_name, id_avg, text[video_name],
                                   [data[p][video_name]['extraversion'],
                                    data[p][video_name]['neuroticism'],
                                    data[p][video_name]['agreeableness'],
                                    data[p][video_name]['conscientiousness'],
                                    data[p][video_name]['openness']]])
            else:
                video_data.append([video_name,
                                   [data[p][video_name]['extraversion'],
                                    data[p][video_name]['neuroticism'],
                                    data[p][video_name]['agreeableness'],
                                    data[p][video_name]['conscientiousness'],
                                    data[p][video_name]['openness']],
                                    text[video_name],
                                    [data[p][video_name]['extraversion'],
                                    data[p][video_name]['neuroticism'],
                                    data[p][video_name]['agreeableness'],
                                    data[p][video_name]['conscientiousness'],
                                    data[p][video_name]['openness']]])

    return video_data

def default_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        return im
    except OSError:
        print(path)
        return Image.new("RGB", (512, 512), "white")

def get_bbox(p, h, w):
    p = np.array(p)
    xmin = min(p[:,0])
    xmax = max(p[:,0])
    ymin = min(p[:,1])
    ymax = max(p[:,1])
    # image size
    xmin = xmin*w
    xmax = xmax*w
    ymin = ymin*h
    ymax = ymax*h
    # expand a little
    long = max(ymax-ymin,xmax-xmin)
    size = long*1.5
    xmin = max(0,xmin-(size-(xmax-xmin))/2)
    xmax = min(w,xmax+(size-(xmax-xmin))/2)
    ymin = max(0,ymin-(size-(ymax-ymin))/2)
    ymax = min(h,ymax+(size-(ymax-ymin))/2)
    # image size
    xmin = int(xmin)
    xmax = int(xmax)
    ymin = int(ymin)
    ymax = int(ymax)
    return xmin,ymin,xmax,ymax

def get_correct_path(root, file_name):
    for old_split in ['train','val','test']:
        path = os.path.join(root, old_split, file_name)
        if os.path.exists(path):
            return path

class ImagerLoader(data.Dataset):
    def __init__(self, frame_root, audio_root,
                label_path, text_path, split,
                average_mode = 'avg',
                transform=None, transform_aug=None,
                video_len=30, stride=15,
                head_box_root=None):
        self.frame_root = frame_root
        self.audio_root = audio_root
        self.data = make_dataset(label_path, text_path, split, average_mode)
        self.average_mode = average_mode
        self.transform = transform
        self.transform_aug = transform_aug
        self.video_len = video_len
        self.stride = stride
        self.target_sr = 16000
        self.head_box_root = head_box_root
        self.d_a = 1690

        self.tokenizer = BertTokenizer.from_pretrained('/sunyunjia/tools/transformers_models/bert-base-uncased')
        self.audio_transform = torchaudio.transforms.MelSpectrogram(self.target_sr, n_fft=800, n_mels=90)

    def __getitem__(self,index):
        file_name, label, text, org_label = self.data[index]

        # get head box dict
        if not self.head_box_root is None:
            with open(get_correct_path(self.head_box_root,f'{file_name}.json'),'r') as f:
                head_boxes = json.load(f)

        # folder that contains all frames of a video named <file_name>
        file_path = get_correct_path(self.frame_root,file_name)
        video_path = file_path
        file_len = len(os.listdir(file_path))
        # get image sequences
        video = torch.zeros(self.video_len,3,224,224)
        video_head = torch.zeros(self.video_len,3,224,224)
        for i in range(min(self.video_len,file_len//(self.stride//5))):
            image_name = str(i*self.stride+1).zfill(5)+'.png'
            image_path = get_correct_path(self.frame_root,os.path.join(file_path,image_name))
            image = default_loader(image_path)
            # head image
            if (not self.head_box_root is None) and (image_name in head_boxes):
                landmark = head_boxes[image_name]
                bbox = get_bbox(landmark,image.size[1],image.size[0])
                image_head = image.crop(bbox)
            else:
                image_head = image
            if self.transform_aug and random.random() < 0.5:
                image = self.transform_aug(image)
            else:
                image = self.transform(image)
            video[i] = image
            if self.transform_aug and random.random() < 0.5:
                image_head = self.transform_aug(image_head)
            else:
                image_head = self.transform(image_head)
            video_head[i] = image_head

        # get audio
        file_path = get_correct_path(self.audio_root,file_name[:-4]+'.wav')
        waveform, sr = torchaudio.load(file_path)
        if sr != self.target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)(waveform)
        waveform = waveform[0] + waveform[1]
        mel_spectrogram = self.audio_transform(waveform)
        log_mel = torch.log1p(mel_spectrogram)
        mean = log_mel.mean()
        std = log_mel.std()
        # shape [90,x]
        log_mel = (log_mel - mean) / (std + 1e-5)
        if log_mel.shape[1] >= self.d_a:
            log_mel = log_mel[:,:self.d_a]
        else:
            log_mel = F.pad(log_mel, (0, self.d_a-log_mel.shape[1]))

        # text embedding
        token = self.tokenizer.encode(text)
        token = torch.tensor(token)
        # fixed sequence length   
        attention_mask = F.pad(torch.ones(len(token)),(0, 512-len(token)))
        token = F.pad(token, (0, 512-len(token)))

        label = torch.FloatTensor(label)
        org_label = torch.FloatTensor(org_label)

        sample = {
                # 3,224,224
                'video':video,
                # 3,224,224
                'head':video_head,
                # 90,x
                'audio':log_mel,
                # x
                'text':token.int(),
                'att_mask':attention_mask.int(),
                'label':label,
                'org_label':org_label,
                'video_path':video_path}

        return sample

    def __len__(self):
        return len(self.data)
