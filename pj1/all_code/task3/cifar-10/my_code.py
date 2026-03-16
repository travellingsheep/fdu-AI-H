import numpy as np
import os
import torch
import pickle
from torch import nn
from torchvision import transforms
from torch.utils.data import random_split,Dataset,DataLoader,ConcatDataset
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
        self.conv=basic_conv(3,16)
        self.bn=nn.BatchNorm2d(16)
        self.relu=nn.ReLU()
        self.pool=nn.MaxPool2d(2,2)
        self.layer1=self.make_layer(block,16,layer_num[0])
        self.layer2=self.make_layer(block,32,layer_num[1],stride=2)
        self.layer3=self.make_layer(block,64,layer_num[2],stride=2)
        self.average_pool=nn.AvgPool2d(8)
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
        test_dataset=format_changer([eval_datafile_path],transform)
        test_loader=DataLoader(test_dataset,batch_size=64,shuffle=True)
        self.my_test(device,test_loader)

if __name__ == "__main__":
    path=r"cifar-10-batches-py"
    train_file=[os.path.join(path,f"data_batch_{i}")for i in range(1,6)]
    transform=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
    train_dataset=ConcatDataset([format_changer([file],transform=transform) for file in train_file])

    train_set_size=int(0.8*len(train_dataset))
    val_set_ize=len(train_dataset)-train_set_size
    train_set,val_set=random_split(train_dataset,[train_set_size,val_set_ize])
    
    #train and evaluate
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=ResNet(BasicBlock,[3,3,3]).to(device)
    optimizer=torch.optim.SGD(model.parameters(),lr=0.1,weight_decay=0.0001,momentum=0.9)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[32000,48000])
    train_loader=DataLoader(train_set,batch_size=128,shuffle=True)
    val_loader=DataLoader(val_set,batch_size=128,shuffle=True)

    xlabel=[]
    test_acc=[]
    iteration=0
    for epoch_th in range(120):
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

    with open("pj3_2_4.pickle","wb") as f:
        pickle.dump(model,f)