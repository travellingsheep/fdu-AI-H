import pickle
from mymodel import format_changer,BasicBlock,ResNet
import pickle
import torch
from torchvision import transforms

if __name__=="__main__":
    model_path=r".pickle"
    eval_datafile_path=[r"t10k-images-idx3-ubyte",r"t10k-labels-idx1-ubyte"]
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
    with open(model_path,"rb") as f:
        model=pickle.load(f)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.interview(device,eval_datafile_path,transform)