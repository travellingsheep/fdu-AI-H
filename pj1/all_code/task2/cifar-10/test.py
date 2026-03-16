import pickle
import os
from mymodel import format_changer,ConvNN
import pickle
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

if __name__=="__main__":
    model_path=r".pickle"
    eval_datafile_dir=r"cifar-10-batches-py"
    eval_datafile_path=os.path.join(eval_datafile_dir,"test_batch")
    transform=transforms.Compose([transforms.ToTensor()])
    #test_set=format_changer([eval_datafile_path],transform)
    with open(model_path,"rb") as f:
        model=pickle.load(f)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.interview(device,eval_datafile_path,transform)