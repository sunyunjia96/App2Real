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

def get_bbox(b, h, w):
    xmin = b[0]
    xmax = b[2]
    ymin = b[1]
    ymax = b[3]
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

def get_minmax(p):
    p = np.array(p)
    xmin = min(p[:,0])
    xmax = max(p[:,0])
    ymin = min(p[:,1])
    ymax = max(p[:,1])
    return [xmin,ymin,xmax,ymax]

# for TEST ONLY
# transcript file as an anchor
# one id one batch of samples?
# output: [[id, frame folder path, audio path, [[s,e],...], [<text>,...], label, original bbox],...]
# id: '<group>_<letter>'
# s, e in seconds
# fps=30
# original bbox: maxs and mins, get real bbox in dataloader
def make_dataset(frame_root, audio_root, transcript_path, label_file, pos_file, landmark_root):
    data = []
    # only use data over <interval> seconds
    interval = 15
    with open(transcript_path,'r') as f:
        transcript = json.load(f)
    with open(label_file,'r') as f:
        label_dict = json.load(f)
    with open(pos_file,'r') as f:
        pos_corr = json.load(f)

    for group in transcript:
        for letter in transcript[group]:
            # choose intervals
            used_interval = []
            texts = []
            for speak in transcript[group][letter]:
                if speak[0][1]-speak[0][0] >= interval:
                    used_interval.append(speak[0])
                    texts.append(speak[1])
            if len(used_interval)==0:
                continue

            # id
            pid = group+'_'+letter
            # audio
            audio_path = os.path.join(audio_root,group+'.wav')
            # get video id
            if letter in pos_corr[group+'_1']:
                video_path = os.path.join(frame_root, group+'_1')
                ldmk_path = os.path.join(landmark_root, group+'_1.json')
            elif letter in pos_corr[group+'_2']:
                video_path = os.path.join(frame_root, group+'_2')
                ldmk_path = os.path.join(landmark_root, group+'_2.json')
            # head bboxes
            with open(ldmk_path,'r') as f:
                landmarks = json.load(f)
            bbox_dict = {}
            for frame_idx in landmarks:
                if landmarks[frame_idx]:
                    if letter in landmarks[frame_idx]:
                        bbox_dict[frame_idx] = get_minmax(landmarks[frame_idx][letter])
            # label in ENACO
            label = label_dict[group[5:]][letter]
            
            data.append([pid, video_path, audio_path, used_interval, texts, label,bbox_dict])

            

    return data

def default_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        return im
    except OSError:
        print(path)
        return Image.new("RGB", (512, 512), "white")
    
class ImagerLoader(data.Dataset):
    def __init__(self, frame_root, audio_root,
                transcript_path, label_file,
                pos_file, landmark_root,
                audio_delay_file,
                transform=None,
                video_len=30, stride=15):
        
        delay_dict = {}
        with open(audio_delay_file,'r') as f:
            for line in f:
                delay_data = line.strip().split()
                delay_dict[delay_data[0]] = int(float(delay_data[1])/1000)
        self.delay_data = delay_dict

        self.data = make_dataset(frame_root, audio_root,
                                 transcript_path, label_file,
                                 pos_file, landmark_root)
        self.transform = transform
        self.video_len = video_len
        self.stride = stride
        self.target_sr = 16000
        self.d_a = 1690
        self.choose_s = [1,0,-1,-2,2]

        self.tokenizer = BertTokenizer.from_pretrained('/sunyunjia/tools/transformers_models/bert-base-uncased')
        self.audio_transform = torchaudio.transforms.MelSpectrogram(self.target_sr, n_fft=800, n_mels=90)

    def __getitem__(self,index):
        pid, video_path, audio_path, used_interval, texts, label, bbox_dict = self.data[index]
        if pid.split('_')[0][-2:] in self.delay_data:
            delay = self.delay_data[pid.split('_')[0][-2:]]
        else:
            delay = 0

        videos = []
        video_heads = []
        log_mels = []
        tokens = []
        att_masks = []
        for chunk_idx, chunk in enumerate(used_interval):
            fps = 30
            # get image sequences
            video = torch.zeros(self.video_len,3,224,224)
            video_head = torch.zeros(self.video_len,3,224,224)
            for i in range(min(self.video_len,int((chunk[1]-chunk[0])*(fps/self.stride)))):
                image_name = str(i*self.stride + int((chunk[0]+delay)*fps) + self.choose_s[int(chunk[0]*fps)%5]).zfill(5)+'.png'
                image_path = os.path.join(video_path,image_name)
                if not os.path.exists(image_path):
                    break
                image = default_loader(image_path)
                # head image
                if image_name in bbox_dict:
                    landmark = bbox_dict[image_name]
                    bbox = get_bbox(landmark,image.size[1],image.size[0])
                    image_head = image.crop(bbox)
                else:
                    image_head = image
                image = self.transform(image)
                video[i] = image
                image_head = self.transform(image_head)
                video_head[i] = image_head

            # get audio
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.target_sr:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)(waveform)
            # corresponding chunk of audio
            start_sample = int(chunk[0] * self.target_sr)
            num_samples_15s = 15 * self.target_sr
            end_sample = min(start_sample + num_samples_15s, waveform.shape[1])
            waveform = waveform[:, start_sample:end_sample]
            waveform = waveform[0]*2
            # transform
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
            token = self.tokenizer.encode(texts[chunk_idx])
            token = torch.tensor(token)
            # fixed sequence length   
            attention_mask = F.pad(torch.ones(len(token)),(0, 512-len(token)))
            token = F.pad(token, (0, 512-len(token)))

            videos.append(video.unsqueeze(0))
            video_heads.append(video_head.unsqueeze(0))
            log_mels.append(log_mel.unsqueeze(0))
            tokens.append(token.int().unsqueeze(0))
            att_masks.append(attention_mask.int().unsqueeze(0))

        label = torch.FloatTensor(label)
        label = (label-1)/4

        
        sample = {
                # 3,224,224
                'video':torch.cat(videos,dim=0),
                # 3,224,224
                'head':torch.cat(video_heads,dim=0),
                # 90,x
                'audio':torch.cat(log_mels,dim=0),
                # x
                'text':torch.cat(tokens,dim=0),
                'att_mask':torch.cat(att_masks,dim=0),
                'label':label,
                'pid':pid}

        return sample

    def __len__(self):
        return len(self.data)
