

import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
from PIL import Image
import os
import h5py
import numpy as np
import random
from argparse import ArgumentParser
import sys
import pandas as pd
from backbones.CNN3D import *
from backbones.data_loader import *
import pickle

class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, videos_dir, video_names, score, video_format='RGB', width=None, height=None):

        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names
       # self.score = video_names
        self.format = video_format
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx].split("--")[0]
        
        video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name))
      
        video_data2 = video_data.astype(np.float32)

        dliated_data = make_dliated(video_data2)
       # print(video_data.shape,dliated_data)
   
        
        
        video_score = self.video_names[idx].split("--")[1]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        video_length = video_data.shape[0]
        if video_length >8000:
            return
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        transformed_video = torch.zeros([video_length, video_channel,  video_height, video_width])
        
    
        for frame_idx in range(video_length):
            frame = video_data[frame_idx]
            frame = Image.fromarray(frame)
            frame = transform(frame)
            transformed_video[frame_idx] = frame
            
#         for frame_idx in range(video_length):
#             frame = dliated_data[frame_idx]
#             frame = Image.fromarray(frame)
#             frame = transform(frame)
#             transformed_dliated_data[frame_idx] = frame

        dliated_data = dliated_data.repeat(3,axis=3)
#         print(dliated_data.shape)
#         sys.exit()
        dliated_data = torch.from_numpy(dliated_data)
       
    
        dliated_data = dliated_data.permute(0,3,1,2).float()
        sample = {'video': [transformed_video,dliated_data],
                  'score': video_score,
                  "name":video_name,
                 }
      #  print(sample)
      #  sys.exit()
        return sample


class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""
    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        # features@: 7->res5c
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
                features_std = global_std_pool2d(x)
                return features_mean, features_std


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


def get_features(video_data,extractor, frame_batch_size=64, device='cuda'):
    """feature extraction"""
   # extractor = ResNet50().to(device)
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    
    

    output5 = torch.Tensor().to(device)
    
    
    
    extractor.eval()
    with torch.no_grad():
	    while frame_end < video_length:
	        batch = video_data[frame_start:frame_end].to(device)
	        features_mean, features_std = extractor(batch)
	        output1 = torch.cat((output1, features_mean), 0)
	        output2 = torch.cat((output2, features_std), 0)
	  
            
	        frame_end += frame_batch_size
	        frame_start += frame_batch_size
        
	    last_batch = video_data[frame_start:video_length].to(device)
	    features_mean, features_std= extractor(last_batch)
	    output1 = torch.cat((output1, features_mean), 0)
	    output2 = torch.cat((output2, features_std), 0)
        
            
	    output = torch.cat((output1, output2), 1).squeeze()
        
        

    return output


if __name__ == "__main__":
    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--database', default='MY-dataset', type=str,
                        help='database name (default: KoNViD-1k)')
    parser.add_argument('--frame_batch_size', type=int, default=64,
                        help='frame batch size for feature extraction (default: 64)')

    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True



#     videos_dir = "/cfs/cfs-3cab91f9f/liuzhang/video_data/video_clarity_vid2"
#     videos_dir2 = "/cfs/cfs-3cab91f9f/liuzhang/video_data/video_clarity_vid"
#     features_dir = "/cfs/cfs-3cab91f9f/liuzhang/video_data/CNN_features_mydata2/"   
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    
    base_path = "/cfs/cfs-3cab91f9f/liuzhang/open_datasets/ft_local/cvd2014_split/"
    train_path="/cfs/cfs-3cab91f9f/liuzhang/open_datasets/ft_local/cvd2014_split_label_train.pickle"
    val_path ="/cfs/cfs-3cab91f9f/liuzhang/open_datasets/ft_local/cvd2014_split_label_val.pickle" 
    
    features_dir = "/cfs/cfs-3cab91f9f/liuzhang/open_datasets/ft_local/cvd2014_val_features_O/"
    features_dir2 = "/cfs/cfs-3cab91f9f/liuzhang/open_datasets/ft_local/cvd2014_val_features_D/"
    
    
    
    with open(val_path,"rb") as f:
        train_data = pickle.load(f)
        
        
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    if not os.path.exists(features_dir2):
        os.makedirs(features_dir2)   
    down = os.listdir(features_dir)
    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    

    
    extractor = ResNet50().to(device)
    
    video_name_score = []

        
        
    for name1,score in train_data.items():
        if 1:
                name = name1 +"--"+str(score)
                
                video_name_score.append(name)
            
    print(len(video_name_score))

    #print(video_name_score[-20:])
    
 #   sys.exit()

    video_format = "RGB"
    dataset = VideoDataset(base_path, video_name_score, [], video_format,)
    max_len = 0
    print(len(dataset))

    
    for i in range(len(dataset)):
        current_data = dataset[i]
        if not current_data:
            print(i)
            continue
        current_video,dliated_data = current_data['video']
        current_score = current_data['score']
        current_name = current_data["name"]
        
        current_name = current_name.split("/")[-1]
        
        
        print('Video {}: length {} name {} socre {}'.format(i, current_video.shape[0],current_name,current_score))
        #max_len=max(max_len,current_video.shape[0])
        features = get_features(current_video,extractor, args.frame_batch_size, device)
#        print(len(features))
        print("start dliated_data")
        features2 = get_features(dliated_data,extractor, args.frame_batch_size, device)
        
        #sys.exit()
        np.save(features_dir + current_name + '_origin_res5c'+"--"+str(current_score), features.cpu())
        
        np.save(features_dir2 + current_name + '_dliated_res5c'+"--"+str(current_score), features.cpu())
    
        
        #np.save(features_dir + current_name + '_score', current_score)
        
