

from argparse import ArgumentParser
import os
import h5py
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
from scipy import stats
from tensorboardX import SummaryWriter
import datetime
import pandas as pd
import sys
from tqdm import tqdm
from backbones.CNN3D import *
from backbones.data_loader import *



writer = SummaryWriter('runs/exp')
# class VQADataset(Dataset):
#     def __init__(self, features_dir3, index=None, max_len=8000, feat_dim=4096, scale=1):
#         super(VQADataset, self).__init__()
#         self.folders = index
#         self.features_dir = features_dir3
#         self.max_len = max_len
#         self.feat_dim = feat_dim
#         self.scale = scale
# #         self.features = np.zeros((len(index), max_len, feat_dim))
# #         self.length = np.zeros((len(index), 1))
# #         self.mos = np.zeros((len(index), 1))
# #         for i in range(len(index)):
# #             features = np.load(features_dir + index[i] + '_resnet-50_res5c.npy')
# #             if features.shape[0] > max_len:
# #                 features = features[0:max_len,:]
# #             self.length[i] = features.shape[0]
            
            
            
# #             self.features[i, :features.shape[0], :] = features
# #             self.mos[i] = np.load(features_dir + index[i] + '_score.npy')  #
            
        
# #         self.scale = scale  #
# #         self.label = self.mos / self.scale  # label normalization
#       #  print(self.features.shape,self.length.shape,self.label.shape)
#     def __len__(self):
#         return len(self.folders)

    
#     def get_img(self,path):
        
#        # data = np.zeros((max_len, self.feat_dim))
#         features = np.load(self.features_dir + path)
#         if features.shape[0] > max_len:
#             features = features[0:max_len,:]
#         length = features.shape[0]
#         label = int(path.split("--")[1][0])
#      #   data[:length,:] = features
#         name = path.split("_")[0]
#         return features,length,label,name
        
        
#     def __getitem__(self, idx):
        
#         img_data,length,label,name= self.get_img(self.folders[idx])
        
        
        
        
#         sample = img_data,length,label,name
#         return sample





if __name__ == "__main__":
    parser = ArgumentParser(description='"VSFA: Quality Assessment of In-the-Wild Videos')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='number of epochs to train (default: 2000)')


    parser.add_argument('--model', default='VSFA', type=str,
                        help='model name (default: VSFA)')


    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0.0)')


    args = parser.parse_args()

    args.decay_interval = int(args.epochs/100)
    args.decay_ratio = 0.8

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True




    num_for_val = 4


    features_dir = "/cfs/cfs-3cab91f9f/liuzhang/video_data/CNN_features_mydata2/" 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features_dir2 = "/cfs/cfs-3cab91f9f/liuzhang/video_data/CNN_features_mydata3/" 
    
    
    bad_tmp = "./data/bad.csv"
    
    bad_data = pd.read_csv(bad_tmp)
    bad_video = list(bad_data["vid"])

    epoch_start=0
    
    
    
    
    # 训练数据集合
    videos_pic1 = []
    result = os.listdir(features_dir)        
    
    
    total_videos = len(result)
    print(total_videos)
    
    
    width = height=0
    max_len = 8000
    train_list,val_list,test_list =[],[],[]
    best_acc =0
    
    print("训练数据目录：",features_dir)

    
    for i in range(total_videos):
        tmp = result[i].split(".")[0]
        if tmp in bad_video:
            continue

        train_list.append(result[i])
        
        

    print("val数据目录：",features_dir2)
    
    result2 = os.listdir(features_dir2)  
    
    for i in range(len(result2)):
        tmp = result2[i].split(".")[0]
        val_list.append(result2[i])
        
        
     
    print("split data:train: {}, test: {}, val: {}".format(len(train_list),len(test_list),len(val_list)))


    #print(train_list[0])

    
#  #   print(len(train_index))
#     train_list = train_list[0:100]
#     val_list =  val_list[0:10]
    
    
    train_dataset = VQADataset(features_dir,train_list, max_len)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=2)
    
    print("load train data success!")
#     for i, (features, length, label,name) in enumerate(train_loader):
#         print(features.shape,length.shape)
#         break
        
    

    
    val_dataset = VQADataset(features_dir2, val_list, max_len)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,)

    print("load val data success!")
    
    
    model = ResNet3D(Bottleneck, [3, 4, 6, 3]).to(device)  #
    
    

    if not os.path.exists('models'):
        os.makedirs('models')
    trained_model_file = 'models'
    
    
#     if torch.cuda.device_count() > 1:
#         print("Using", torch.cuda.device_count(), "GPUs!")
#         model = nn.DataParallel(model)
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    

    pretrained =1
    if pretrained:
        new_state_dict = {}
        path = "./models/3D_VSFA1_acc:0.43177764565992865.pth"
        checkpoint = torch.load(path)
        
     #   model.load_state_dict(checkpoint)
        for k, v in checkpoint["model"].items():
            name =k # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。
        model.load_state_dict(new_state_dict)
        epoch_start = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("pre_modle:{}".format(path))
        
 
    
    Not_well_video ={}
    
    
    criterion = nn.L1Loss()  # MSELoss loss
   
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)
    
    for epoch in range(epoch_start,args.epochs):
        # Train
        model.train()
        L = 0
        right_num = 0
        total_num = 0
        print("training epch:{},total epch:{}".format(epoch,args.epochs))
        for i, (features, length, label,name) in enumerate(tqdm(train_loader)):
            
#             print("features,",features.shape,)
#             print("length",length)
            features = features.to(device).float()
            label = label.to(device).float()
            optimizer.zero_grad()  #
            
            outputs = model(features)
         #   print(outputs.shape,label.shape)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            L = L + loss.item()
            right = outputs==label
            right_num +=right
        
            
        train_loss = L / (i + 1)
      
        print("train_loss:",train_loss)
 
        if epoch % num_for_val ==0:
            print("start valling")
            model.eval()
            # Val
            y_pred = np.zeros(len(val_list))
            y_val = np.zeros(len(val_list))
            L = 0
            with torch.no_grad():
                badcase = {}
          

               # y_pred = np.zeros(len(result))
             #   y_test = np.zeros(len(result))
                L = 0
                for i, (features, length, label,name) in enumerate(tqdm(val_loader)):
      
                    y_val[i] = 1 * label.item() 
                    features = features.to(device).float()
                    label = label.to(device).float()
                    outputs = model(features)
                    if outputs >1.5:
                        y_pred[i] =2
                    elif outputs <0.5:
                        y_pred[i] = 0
                    else:
                        y_pred[i] =1
                    if y_pred[i] != label:
                        badcase[name[0]] = [float(outputs),int(label)]
                    #y_pred[i] = 1 * outputs.item()
                    loss = criterion(outputs, label)
                    #print(outputs,label)
                    L = L + loss.item()

            re = y_pred == y_val
            
            
            acc = sum(re)/len(val_list)
            print("Acc:",acc)
            val_loss = L / (i + 1)

            #print("Badcase",badcase)
            if acc >best_acc:


                print("save model at epch")

                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state,os.path.join(trained_model_file,"3D_VSFA{}_acc:{}.pth".format(epoch+1,acc)))
                print("Epoch {} model saved!".format(epoch + 1))
                best_acc =acc
            for name,_ in badcase.items():
                if name not in Not_well_video:
                    Not_well_video[name] = 1
                else:
                    Not_well_video[name] +=1
        if 0:
            delect = {}
            for name,time in Not_well_video.items():
                if time >=3:
                    delect[name] = ["not good",time]
            
            df3 = pd.DataFrame.from_dict(delect, orient='index',columns=['score',"label"])
            df3 = df3.reset_index().rename(columns = {'index':'vid'})    
            df3.to_csv("./data/not_good.csv",index=0,model="a",header=None)   
            print("split_t_v {} video".format(len(delect)))
            Not_well_video = {}


#         print(badcase)
          
