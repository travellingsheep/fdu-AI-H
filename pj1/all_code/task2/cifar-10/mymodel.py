import numpy as np
import pickle
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt

#define data loader
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#change the format of data
class format_changer(Dataset):
    def __init__(self,path,transform):
        self.image=[]
        self.labels=[]
        self.transform=transform
        for file in path:
            temp=unpickle(file)
            self.image.append(temp[b'data'])
            self.labels+=temp[b'labels']
        self.image=np.concatenate(self.image,axis=0).reshape(-1,3,32,32)
        self.image=self.image.transpose((0,2,3,1))
        self.labels=np.array(self.labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,index):
        image_temp=self.image[index]
        label_temp=self.labels[index]
        image_temp=transforms.ToPILImage()(image_temp)
        if self.transform:
            image_temp=self.transform(image_temp)
        return image_temp,label_temp

##define the nn
class ConvNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu=nn.ReLU()
        self.pool=nn.MaxPool2d(2,2)
        self.average_pool=nn.AvgPool2d(2,2) 
        self.conv1=nn.Conv2d(3,16,3,padding=1,bias=False) #in: 3 channel, out: 16 channels, size=32-3+1+1*2=32
        self.bn1=nn.BatchNorm2d(16)
        #relu&pool:32/2=16
        self.conv2=nn.Conv2d(16,32,3,padding=1,bias=False) #in: 16 channels, out:32 channels, size=16
        self.bn2=nn.BatchNorm2d(32)
        #relu&pool:16/2=8
        self.conv3=nn.Conv2d(32,64,3,padding=1,bias=False) #in: 32 channels, out:64 channels, size=8
        self.bn3=nn.BatchNorm2d(64)
        #relu&pool:8/2=4
        self.conv4=nn.Conv2d(64,128,3,padding=1,bias=False) #in: 64 channels, out:128 channels, size=4
        self.bn4=nn.BatchNorm2d(128)
        #flatten:128*4*4->2048
        self.fc1=nn.Linear(2048,128) #in:vector[1024], out:vector[512]
        #self.dp1=nn.Dropout(0.5)
        self.fc2=nn.Linear(128,32) #in:vector[512], out:vector[120]
        #self.dp2=nn.Dropout(0.5)
        self.fc3=nn.Linear(32,10) #in:vector[120], out:vector[10]
    
    def forward(self,image):
        temp=self.conv1(image)
        temp=self.bn1(temp)
        temp=self.relu(temp)
        temp=self.pool(temp)

        temp=self.conv2(temp)
        temp=self.bn2(temp)
        temp=self.relu(temp)
        temp=self.pool(temp)

        temp=self.conv3(temp)
        temp=self.bn3(temp)
        temp=self.relu(temp)
        temp=self.average_pool(temp)

        temp=self.conv4(temp)
        temp=self.bn4(temp)
        temp=self.relu(temp)
        #temp=self.pool(temp)

        temp=temp.view(temp.size(0),-1)

        temp=self.fc1(temp)
        temp=self.relu(temp)
        #temp=self.dp1(temp)
        temp=self.fc2(temp)
        temp=self.relu(temp)
        #temp=self.dp2(temp)
        temp=self.fc3(temp)
        return temp

    def my_train(self,device,train_loader,optimizer):
        self.train()
        for epoch,(image,label) in enumerate(train_loader):
            image,label=image.to(device),label.to(device)
            optimizer.zero_grad()
            result=self(image)
            loss=nn.functional.cross_entropy(result,label)
            loss.backward()
            optimizer.step()
            if epoch%20==0:
                print(f"epoch {epoch}: train loss = {loss:.6f}")

    def my_test(self,device,test_loader):
        self.eval()
        val_loss=0
        correct=0
        with torch.no_grad():
            for image,label in test_loader:
                image,label=image.to(device),label.to(device)
                result=self(image)
                val_loss+=nn.functional.cross_entropy(result,label,reduction="sum").item()
                _,prediction=result.max(1,keepdim=True)
                correct+=prediction.eq(label.view_as(prediction)).sum().item()
        val_loss/=len(test_loader.dataset) #test_loader.dataset is the length of the entire dataset
        accuracy=100*correct/len(test_loader.dataset)
        print(f"val loss = {val_loss:.6f}, accuracy={accuracy:.2f}%")
        return accuracy
    
    def interview(self,device,eval_datafile_path,transform):
        test_dataset=format_changer([eval_datafile_path],transform)
        test_loader=DataLoader(test_dataset,batch_size=64,shuffle=True)
        self.my_test(device,test_loader)