import numpy as np
import os
import struct
import torch
import pickle
from torch import nn
from torchvision import transforms
from torch.utils.data import random_split,Dataset,DataLoader
import matplotlib.pyplot as plt

#change data format
class format_changer(Dataset):
    def __init__(self,image_path,label_path,transform):
        self.image_dataset=self.read_image_dataset(image_path)
        self.label_dataset=self.read_label_dataset(label_path)
        self.transform=transform
    
    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self,index):
        image=self.image_dataset[index]
        label=self.label_dataset[index]
        if self.transform:
            image=self.transform(image)
        return image,label
    
    def read_image_dataset(self,file_path):
        with open(file_path,'rb') as f:
            magic_num,image_num,row,col=struct.unpack('>IIII',f.read(16))
            data=np.fromfile(f,dtype=np.uint8)
            data=data.reshape(image_num,row,col)
            data=data.astype(np.float32)/255.0
        return data
    
    def read_label_dataset(self,file_path):
        with open(file_path,'rb') as f:
            magic_num,label_num=struct.unpack('>II',f.read(8))
            data=np.fromfile(f,dtype=np.uint8)
            #data=np.eye(10)[data] no need
        return torch.from_numpy(data).long()
    
#define a basic conv
def basic_conv(in_channel,out_channel,stride=1):
    return nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False)

##define the nn
class BasicBlock(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1,downsample=None):
        super().__init__()
        self.relu=nn.ReLU()
        self.conv1=basic_conv(in_channel,out_channel,stride)
        self.bn1=nn.BatchNorm2d(out_channel)
        self.conv2=basic_conv(out_channel,out_channel)
        self.bn2=nn.BatchNorm2d(out_channel)
        self.downsample=downsample

    def forward(self,image):
        residual=image

        temp=self.conv1(image)
        temp=self.bn1(temp)
        temp=self.relu(temp)

        temp=self.conv2(temp)
        temp=self.bn2(temp)

        if self.downsample:
            residual=self.downsample(image)

        temp+=residual
        temp=self.relu(temp)

        return temp

class ResNet(nn.Module):
    def __init__(self,block,layer_num,num_class=10):
        super().__init__()
        self.in_channel=16
        self.conv=basic_conv(1,16) 
        self.bn=nn.BatchNorm2d(16)
        self.relu=nn.ReLU()
        self.pool=nn.MaxPool2d(2,2)
        self.layer1=self.make_layer(block,16,layer_num[0])
        self.layer2=self.make_layer(block,32,layer_num[1],stride=2)
        self.layer3=self.make_layer(block,64,layer_num[2],stride=2)
        self.average_pool=nn.AvgPool2d(4)
        self.dp=nn.Dropout(0.5)
        self.fc=nn.Linear(64,num_class)
    
    def make_layer(self,block,out_channel,block_num,stride=1):
        downsample=None
        if stride!=1 or self.in_channel!=out_channel:
            downsample=nn.Sequential(
                nn.Conv2d(self.in_channel,out_channel,kernel_size=1,stride=stride,padding=0,bias=False),
                nn.BatchNorm2d(out_channel)
            )
        
        layer=[]
        layer.append(block(self.in_channel,out_channel,stride,downsample))
        self.in_channel=out_channel
        for i in range(1,block_num):
            layer.append(block(out_channel,out_channel))
        
        return nn.Sequential(*layer)

    def forward(self,image):
        temp=self.conv(image)
        temp=self.bn(temp)
        temp=self.relu(temp)

        temp=self.layer1(temp)
        temp=self.layer2(temp)
        temp=self.layer3(temp)

        temp=self.average_pool(temp)
        temp=temp.view(temp.size(0),-1)
        temp=self.dp(temp)
        temp=self.fc(temp)

        return temp

    def my_train(self,device,image,label,optimizer):
        self.train()
        image,label=image.to(device),label.to(device)
        optimizer.zero_grad()
        result=self(image)
        loss=nn.functional.cross_entropy(result,label)
        loss.backward()
        optimizer.step()

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
        eval_datafile_path_image=eval_datafile_path[0]
        eval_datafile_path_label=eval_datafile_path[1]
        test_dataset=format_changer(eval_datafile_path_image,eval_datafile_path_label,transform)
        test_loader=DataLoader(test_dataset,batch_size=64,shuffle=True)
        self.my_test(device,test_loader)

if __name__ == "__main__":
    image_path=r"train-images.idx3-ubyte"
    label_path=r"train-labels.idx1-ubyte"
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
    train_dataset=format_changer(image_path,label_path,transform)

    train_set_size=int(0.8*len(train_dataset))
    val_set_ize=len(train_dataset)-train_set_size
    train_set,val_set=random_split(train_dataset,[train_set_size,val_set_ize])
    train_loader=DataLoader(train_set,batch_size=64,shuffle=True) #an iterator which returns a batch of (one image,one label) until the whole dataset is used
    val_loader=DataLoader(val_set,batch_size=64,shuffle=False)
    
    
    #train and evaluate
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=ResNet(BasicBlock,[1,1,1]).to(device)
    optimizer=torch.optim.SGD(model.parameters(),lr=0.01,weight_decay=0.0001,momentum=0.9)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[32000,48000])
    train_loader=DataLoader(train_set,batch_size=128,shuffle=True)
    val_loader=DataLoader(val_set,batch_size=128,shuffle=True)

    xlabel=[]
    test_acc=[]
    iteration=0
    for epoch_th in range(30):
        xlabel.append(epoch_th+1)
        print(f"epoch:{epoch_th}")
        for image,label in train_loader:
            model.my_train(device,image,label,optimizer)
            iteration+=1
            if iteration==32000:
                print(f"lr division 1")
            if iteration==48000:
                print(f"lr division 2")
            scheduler.step()
        test_acc_temp=model.my_test(device,val_loader)
        test_acc.append(test_acc_temp)

    plt.plot(xlabel,test_acc,label="test accuracy")
    plt.legend()
    plt.show()

    with open("pj3_1_1.pickle","wb") as f:
        pickle.dump(model,f)