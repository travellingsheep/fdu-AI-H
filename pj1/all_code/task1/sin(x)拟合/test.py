import pickle
from mymodel import FC
import matplotlib.pyplot as plt
import numpy as np
import random

if __name__=="__main__":
    with open(r".pickle","rb") as f:
        model=pickle.load(f)
    
    test_input=np.linspace(-3.14,3.14,1000).reshape(-1,1)
    test_arr=np.hstack((test_input,np.zeros((1000,1))))
    activation=model.forward_propagation(test_arr)
    loss=model.calculate_sampling_error()
    print(loss)
    plt.plot(test_input,activation[-1][0],label="predicted sin(x)")
    plt.plot(test_input,np.sin(test_input),label="true sin(x)")
    plt.legend()
    plt.show()
