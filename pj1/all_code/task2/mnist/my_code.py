import numpy as np
import struct
import pickle
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset,random_split,DataLoader

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
    
##define the nn
class ConvNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,10,5) #in: 1 channel, out: 10 channels, size=28-5+1=24
        self.relu=nn.ReLU()
        self.pool=nn.MaxPool2d(2,2) #in:10*24*24, out:10*12*12
        self.conv2=nn.Conv2d(10,20,3) #in: 10 channels, out:20 channels, size=12-3+1=10
        #flatten:20*5*5->500
        #self.fc1=nn.Linear(2000,500) #in:vector[2000], out:vector[500]
        self.fc1=nn.Linear(500,120) #in:vector[500], out:vector[120]
        self.fc2=nn.Linear(120,10) #in:vector[500], out:vector[120]
    
    def forward(self,image):
        image=self.conv1(image)
        image=self.relu(image)
        image=self.pool(image)
        image=self.conv2(image)
        image=self.relu(image)
        image=self.pool(image)
        image=image.view(-1,500)
        image=self.fc1(image)
        image=self.relu(image)
        image=self.fc2(image)
        #image=nn.functional.log_softmax(image):no need, as nn.functional.cross-entropy has contained the log function
        return image

    def my_train(self,device,train_loader,optimizer,epoch_num):
        self.train()
        for epoch,(image,label) in enumerate(train_loader):
            image,label=image.to(device),label.to(device)
            optimizer.zero_grad()
            result=self(image)
            loss=nn.functional.cross_entropy(result,label)
            loss.backward()
            optimizer.step()
            if (epoch)%20==0 or epoch==epoch_num-1:
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
    
    def interview(self,device,eval_datafile_path,transform):
        eval_datafile_path_image=eval_datafile_path[0]
        eval_datafile_path_label=eval_datafile_path[1]
        test_dataset=format_changer(eval_datafile_path_image,eval_datafile_path_label,transform)
        test_loader=DataLoader(test_dataset,batch_size=64,shuffle=True)
        self.my_test(device,test_loader)

if __name__ == "__main__":
    image_path=r"train-images.idx3-ubyte"
    label_path=r"train-labels.idx1-ubyte"
    ##load data and convert into iterator
    #set=((image,label)*n),image is only one image, label is only one label
    #loader=((image,label)),image is a batch of images,size=(batch_size,channel_num,row,col)
                           #label is a batch of labels,size=(batch_size)
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
    train_dataset=format_changer(image_path,label_path,transform)
    train_set_size=int(0.8*len(train_dataset))
    val_set_ize=len(train_dataset)-train_set_size
    train_set,val_set=random_split(train_dataset,[train_set_size,val_set_ize])
    train_loader=DataLoader(train_set,batch_size=64,shuffle=True) #an iterator which returns a batch of (one image,one label) until the whole dataset is used
    val_loader=DataLoader(val_set,batch_size=64,shuffle=False)
    ##choose the device and optimizer
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=ConvNN().to(device)
    optimizer=torch.optim.Adam(model.parameters())

    for epoch in range(20):
        print(f"epoch:{epoch}")
        model.my_train(device,train_loader,optimizer,epoch)
        model.my_test(device,val_loader)

    with open("pj2_mnist_res_new_12.pickle","wb") as f:
        pickle.dump(model,f)