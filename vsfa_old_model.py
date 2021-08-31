
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
#from tensorboardX import SummaryWriter
import datetime
import pandas as pd
import sys
from tqdm import tqdm

class VQADataset(Dataset):
    def __init__(self, features_dir='CNN_features_KoNViD-1k/', index=None, max_len=8000, feat_dim=4096, scale=1):
        super(VQADataset, self).__init__()
        self.folders = index
        self.features_dir = features_dir
        self.max_len = 240
        self.feat_dim = feat_dim
        self.scale = 4.05
      #  self.features = np.zeros((len(index), max_len, feat_dim))
#         self.length = np.zeros((len(index), 1))
#         self.mos = np.zeros((len(index), 1))
#         for i in range(len(index)):
#             features = np.load(features_dir + index[i] + '_resnet-50_res5c.npy')
#             if features.shape[0] > max_len:
#                 features = features[0:max_len,:]
#             self.length[i] = features.shape[0]
            
            
            
#             self.features[i, :features.shape[0], :] = features
#             self.mos[i] = np.load(features_dir + index[i] + '_score.npy')  #
            
        
#         self.scale = scale  #
#         self.label = self.mos / self.scale  # label normalization
#        print(self.features.shape,self.length.shape,self.label.shape)
    def __len__(self):
        return len(self.folders)

    
    def get_img(self,path):
        
        data = np.zeros((self.max_len, self.feat_dim))
        features = np.load(self.features_dir + path)
   #     print("old***********************",features.shape)
        if features.shape[0] > self.max_len:
            features = features[0:self.max_len,:]
        length = features.shape[0]
        label = float(path.split("--")[1][0:-4])/self.scale
        data[:length,:] = features
        name = path.split("_")[0]
        return data,length,label,name
        
        
    def __getitem__(self, idx):
        
        img_data,length,label,name= self.get_img(self.folders[idx])
        
        
        
        
        sample = img_data,length,label,name
        return sample


class ANN(nn.Module):
    def __init__(self, input_size=4096, reduced_size=128, n_ANNlayers=1, dropout_p=0.5):
        super(ANN, self).__init__()
        self.n_ANNlayers = n_ANNlayers
    
        self.fc0 = nn.Linear(input_size, reduced_size)  #
        self.dropout = nn.Dropout(p=dropout_p)  #
        self.fc = nn.Linear(reduced_size, reduced_size)  #

    def forward(self, input):
        input = self.fc0(input)  # linear
        for i in range(self.n_ANNlayers-1):  # nonlinear
            input = self.fc(self.dropout(F.relu(input)))
        return input


def TP(q, tau=12, beta=0.5):
    """subjectively-inspired temporal pooling"""
    q = torch.unsqueeze(torch.t(q), 0)
    qm = -float('inf')*torch.ones((1, 1, tau-1)).to(q.device)
    qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)  #
    l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)
    m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)
    n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
    m = m / n
    return beta * m + (1 - beta) * l


class VSFA(nn.Module):
    def __init__(self, input_size=4096, reduced_size=128, hidden_size=32):

        super(VSFA, self).__init__()
        self.hidden_size = hidden_size
        self.ann = ANN(input_size, reduced_size, 1)
        self.rnn = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.q = nn.Linear(hidden_size, 1)

    def forward(self, input, input_length):
       # print(input.shape,input_length.shape)
        input = self.ann(input)  # dimension reduction
       # print(input.shape,input_length.shape)
        self.rnn.flatten_parameters()
        outputs, _ = self.rnn(input, self._get_initial_state(input.size(0), input.device))
      #  print(outputs.shape)
        q = self.q(outputs)  # frame quality
      #  print(q.shape)
        score = torch.zeros_like(input_length, device=q.device)  #
        for i in range(input_length.shape[0]):  #
            qi = q[i, :np.int(input_length[i].cpu().numpy())]
            #print(qi.shape)
            qi = TP(qi)
            score[i] = torch.mean(qi)  # video overall quality
        return score

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0


if __name__ == "__main__":
    parser = ArgumentParser(description='"VSFA: Quality Assessment of In-the-Wild Videos')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='number of epochs to train (default: 2000)')

    parser.add_argument('--database', default='my_dataset', type=str,
                        help='database name (default: CVD2014)')
    parser.add_argument('--model', default='VSFA', type=str,
                        help='model name (default: VSFA)')
    parser.add_argument('--exp_id', default=0, type=int,
                        help='exp id for train-val-test splits (default: 0)')


    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')

    parser.add_argument("--notest_during_training", action='store_true',
                        help='flag whether to test during training')
    parser.add_argument("--disable_visualization", action='store_true',
                        help='flag whether to enable TensorBoard visualization')
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="log directory for Tensorboard log output")

    args = parser.parse_args()

    args.decay_interval = int(args.epochs/10)
    args.decay_ratio = 0.8

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    print('EXP ID: {}'.format(args.exp_id))
    print(args.database)
    print(args.model)
    
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    num_for_val = 20
    epoch_start=0

    features_dir =  "/cfs/cfs-3cab91f9f/liuzhang/open_datasets/ft_local/cvd2014_train_features_O/"
    print("训练数据目录：",features_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features_dir2 = "/cfs/cfs-3cab91f9f/liuzhang/open_datasets/ft_local/cvd2014_val_features_O/"
    
    # 训练数据集合
    videos_pic1 = []
    train_list = os.listdir(features_dir)        
    val_list = os.listdir(features_dir2)
    test_list =[]
#     total_videos = len(result)
#     #print("总数据:",total_videos)
    
    
#     width = height=0
#     max_len = 8000
#     train_list,val_list,test_list =[],[],[]
#     best_acc =0
    
    

    
#     for i in range(total_videos):
#         tmp = result[i].split(".")[0]
#         #rint(tmp)
        
#         if tmp in val_data:
#             val_list.append(result[i])
#             continue

#         train_list.append(result[i])
            
    
        
        
    print(train_list[0],val_list[0])   
    print("split data:train: {}, test: {}, val: {}".format(len(train_list),len(test_list),len(val_list)))
    

    #print(train_list[0])

    
#  #   print(len(train_index))
#     train_list = train_list[0:100]
#     val_list =  val_list[0:10]
    
    
    train_dataset = VQADataset(features_dir,train_list, max_len=240,scale =93.83)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=1)
    
    print("load train data success!")
#     for i, (features, length, label,name) in enumerate(train_loader):
#         print(features.shape,length.shape)
#         break
        
    

    
    val_dataset = VQADataset(features_dir2, val_list, max_len=240,scale = 93.83)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,)

    print("load val data success!")
    
    
    model = VSFA().to(device)  #

    if not os.path.exists('models'):
        os.makedirs('models')
    trained_model_file = 'models'
    
    
#     if torch.cuda.device_count() > 1:
#         print("Using", torch.cuda.device_count(), "GPUs!")
#         model = nn.DataParallel(model)
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_PLCC=0
    pretrained =1
    
    if pretrained:
        new_state_dict = {}
        path = "./models/VSFA.pt"
        checkpoint = torch.load(path)
     #   model.load_state_dict(checkpoint)
        for k, v in checkpoint.items():
            name =k # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。
        model.load_state_dict(new_state_dict)
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         epoch_start = checkpoint['epoch']
#         print("load_success!!!!!!!!!!!!")       
    
    
    
    criterion = nn.L1Loss()  # L1 loss
   
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)
    
    for epoch in range(epoch_start,args.epochs):
        # Train
        model.train()
        L = 0
        print("training epch:{},total epch:{}".format(epoch,args.epochs))
        for i, (features, length, label,name) in enumerate(tqdm(train_loader)):
            
            
            features = features.to(device).float()
            label = label.to(device).float()
            optimizer.zero_grad()  #
            outputs = model(features, length.float())
            #print(outputs.shape,label.shape)
           # outputs= outputs.squeeze(1)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            L = L + loss.item()
            right = outputs==label

        train_loss = L / (i + 1)
        
        
        
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
                for i, (features,length, label,name) in enumerate(tqdm(val_loader)):
      
                    y_val[i] =93.83*label.item()    
                    features = features.to(device).float()
                    label = label.to(device).float()
                
                
                    outputs = model(features,length.float())
                    
                    
                    y_pred[i] =93.83*outputs.item()
      
                    #y_pred[i] = 1 * outputs.item()
                    loss = criterion(outputs, label)
                    #print(outputs,label)
                    L = L + loss.item()
            #print("ypred",y_pred)
           # print("y_label",y_val)
            val_loss = L / (i + 1)
            val_PLCC = stats.pearsonr(y_pred, y_val)[0]
            val_SROCC = stats.spearmanr(y_pred, y_val)[0]
            val_RMSE = np.sqrt(((y_pred-y_val) ** 2).mean())
            val_KROCC = stats.stats.kendalltau(y_pred, y_val)[0]

            #print("Badcase",badcase)
            print("Val results: val loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}" .format(val_loss, val_SROCC, val_KROCC, val_PLCC, val_RMSE))
            if val_PLCC >best_PLCC:


                print("save model at epch")

                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state,os.path.join(trained_model_file,"OLD_VSFG{}_plcc:{}.pth".format(epoch+1,val_PLCC)))
                print("Epoch {} model saved!".format(epoch + 1))
                best_PLCC = val_PLCC
            elif epoch % 20 ==0:
                print("save model at epch")

                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state,os.path.join(trained_model_file,"OLD_VSFG{}_epoch:{}.pth".format(epoch+1,epoch+1)))
                print("Epoch {} model saved!".format(epoch + 1))

#     if test_flag:
        
        
#         check_point = torch.load(os.path.join(trained_model_file,"model_{}.pth".format(epoch+1)))
#         new_state_dict={}
#         for k, v in check_point["model"].items():
#             name =k[7:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
#             new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。
        
        
#         model.load_state_dict(new_state_dict)
#         print("load model {} success".format(pre_model))
        
#         model.eval()
#         with torch.no_grad():
#             badcase = {}
          
          
#             y_pred = np.zeros(len(result))
#             y_test = np.zeros(len(result))
#             L = 0
#             for i, (features, length, label,name) in enumerate(tqdm(test_loader)):
#                 y_test[i] = 1 * label.item() 
#                 features = features.to(device).float()
#                 label = label.to(device).float()
#                 outputs = model(features, length.float())
#                 if outputs >1.5:
#                     y_pred[i] =2
#                 elif outputs <0.1:
#                     y_pred[i] = 0
#                 else:
#                     y_pred[i] =1
#                 if y_pred[i] != label:
#                     badcase[name] = [float(outputs),int(label)]
#                 #y_pred[i] = 1 * outputs.item()
#                 loss = criterion(outputs, label)
#                 L = L + loss.item()
                
                
          
          
#         test_loss = L / (i + 1)
            
#         RMSE = np.sqrt(((y_pred-y_test) ** 2).mean())
        
#         re = y_pred == y_test
#         print(sum(re)/test_num)
#         print(badcase)
          
