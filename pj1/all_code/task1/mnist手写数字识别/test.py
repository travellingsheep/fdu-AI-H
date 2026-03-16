import pickle
from mymodel import FC
import pickle

if __name__=="__main__":
    model_path=r".pickle"
    eval_datafile_path=[r"t10k-images-idx3-ubyte",r"t10k-labels-idx1-ubyte"]
    with open(model_path,"rb") as f:
        model=pickle.load(f)
    acc=model.interview(eval_datafile_path)
    print(f"accuracy is {acc}%")