import matplotlib.pyplot as plt
import numpy as np
import random
import pickle

def initialize_training_dataset(x):
    train_arr=np.zeros((x,2))
    train_arr[:,0] = np.random.uniform(-3.14,3.14,x)
    train_arr[:,1] = np.sin(train_arr[:,0])
    return train_arr

def initialize_validation_dataset(x):
    val_arr=np.zeros((x,2))
    for i in range(x):
        val_arr[i][0]=random.uniform(-3.14,3.14)
        val_arr[i][1]=np.sin(val_arr[i][0])
    return val_arr

class FC:
    def __init__(self,neuron_list,learning_rate):
        self.count=self.initialize_nn_count(neuron_list)
        self.nn=self.initialize_nn_parameters()
        self.learning_rate=learning_rate

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1-np.tanh(x)**2
    
    def initialize_nn_count(self,neuron_list):
        x=len(neuron_list)
        count=np.zeros(x+2,dtype=int)
        for i in range(x):
            count[i+1]=neuron_list[i]
        count[0]=1
        count[x+1]=1
        return count

    def initialize_nn_parameters(self):
        count=self.count
        layer=count.size
        nn=[]
        for i in range(layer-1):
            std=np.sqrt(2.0/(count[i+1]+count[i]))
            weight=np.random.normal(0,std,(count[i+1],count[i]))
            bias = np.random.normal(0, 0.1, (count[i + 1],))
            nn.append((weight,bias))
        return nn
    
    def forward_propagation(self,train_arr):
        nn_parameters=self.nn
        count=self.count
        activation=[train_arr[:,0].reshape(1,-1)]
        layer=count.size
        for i in range(layer-1):
            weight,bias=nn_parameters[i]
            act=np.dot(weight,activation[-1])+bias.reshape(-1,1)
            if i!=layer-2:
                act=self.tanh(act)
            activation.append(act)
        return activation
    
    def initialize_delta_matrix(self):
        count=self.count
        max_neurons=max(count)
        delta_matrix=np.zeros((count.size-1, max_neurons, max_neurons, 2))
        return delta_matrix
    
    def cal_delta_matrix(self,train_arr):
        delta_matrix=self.initialize_delta_matrix()
        count=self.count
        activation=self.forward_propagation(train_arr)
        learning_rate=self.learning_rate
        nn_parameters=self.nn
        train_x=train_arr.shape[0]
        error=np.zeros((count.size,max(count),train_x))
        error[-1,0,:]=train_arr[:,1]-activation[-1][0]
            
        for layer in range(count.size-2,-1,-1):
            weight=nn_parameters[layer][0]
            delta_next=error[layer+1,:count[layer+1],:]
            act=np.array(activation[layer])

            delta_matrix[layer,:count[layer],:count[layer+1],0]=learning_rate*(act@delta_next.T)/train_x
            delta_matrix[layer,0,:count[layer+1],1]=learning_rate*delta_next.sum(axis=1)/train_x
            error[layer,:count[layer],:]=(weight.T@delta_next)*(self.tanh_derivative(act) if layer>0 else 1)

        return delta_matrix
    
    def train_nn(self,train_arr,val_arr,epoch_num,val_delta):
        count=self.count
        nn_parameters=self.nn
        loss=[]
        val_loss=[]
        delta_epoch=[]
        for epoch in range(epoch_num):
            activation=self.forward_propagation(train_arr)
            loss_temp=np.mean((train_arr[:,1]-activation[-1])**2)/2
            loss.append(loss_temp)

            delta_matrix=self.cal_delta_matrix(train_arr)
            for layer in range(count.size-1):
                weight,bias=nn_parameters[layer]
                weight+=delta_matrix[layer,:count[layer],:count[layer+1],0].T
                bias+=delta_matrix[layer,0,:count[layer+1],1]
                nn_parameters[layer]=(weight,bias)
        
            if (epoch)%val_delta==0 or epoch==epoch_num-1:
                val_activation=self.forward_propagation(val_arr)
                val_temp=np.mean((val_arr[:,1]-val_activation[-1])**2)/2
                val_loss.append(val_temp)
                delta_epoch.append(epoch)
                print(f"epoch {epoch}: train loss = {loss_temp:.6f}, val loss = {val_temp:.6f}")

        return loss,val_loss,delta_epoch

    def calculate_sampling_error(self,sample_size=100):
        sample_x=np.random.uniform(-3.14,3.14,sample_size).reshape(-1,1)
        sample_arr=np.hstack((sample_x,np.zeros((sample_size,1))))
        sample_activation=self.forward_propagation(sample_arr)
        predicted=sample_activation[-1][0]
        true_value=np.sin(sample_x.flatten())
        loss=np.mean((predicted-true_value)**2)/2
        return loss
    
    def interview(eval_datafile_path):
        pass

if __name__ == "__main__":
    random.seed()
    train_x=int(input("训练几个？"))
    train_arr=initialize_training_dataset(train_x)
    validation_x=int(input("测试几个？"))
    val_arr=initialize_validation_dataset(validation_x)
    x=int(input("你需要几层（不包括输入输出层）："))
    print(f"你每一层需要几个神经元:")
    if x==1:
        neuron_list=[int(input())]
    else:
        neurons=input().split()
        neuron_list=[int(neuron) for neuron in neurons]
    learning_rate=float(input("你想要的学习率："))
    epoch=int(input("你想要训练多少轮："))

    model=FC(neuron_list,learning_rate)
    loss,val_loss,delta_epoch=model.train_nn(train_arr,val_arr,epoch,100)
    sample_loss=model.calculate_sampling_error()
    print(f"loss: {sample_loss:.6f}")

    plt.plot(range(epoch),loss,label="training loss")
    plt.plot(delta_epoch,val_loss,'o-',label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    
    test_input=np.linspace(-3.14,3.14,1000).reshape(-1,1)
    test_arr=np.hstack((test_input,np.zeros((1000,1))))
    test_activation=model.forward_propagation(test_arr)
    plt.plot(test_input,test_activation[-1][0],label="predicted sin(x)")
    plt.plot(test_input,np.sin(test_input),label="true sin(x)")
    plt.legend()
    plt.show()

    with open("pj1_1.pickle","wb") as f:
        pickle.dump(model,f)