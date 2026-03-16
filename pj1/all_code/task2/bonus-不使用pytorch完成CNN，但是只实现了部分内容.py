import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

###dataset
##read dataset
#read image dataset
def read_image_dataset(file_path):
    with open(file_path,'rb') as f:
        magic_num,image_num,row,col=struct.unpack('>IIII',f.read(16))
        data=np.fromfile(f,dtype=np.uint8)
        data=data.reshape(image_num,row,col)
        data=data.astype(np.float32)/255.0
    return data,image_num,row

#read label dataset
def read_label_dataset(file_path):
    with open(file_path,'rb') as f:
        magic_num,label_num=struct.unpack('>II',f.read(8))
        data=np.fromfile(f,dtype=np.uint8)
        data=np.eye(10)[data]
    return data

##split dataset
#split image and label, with specific ratio between 0 and 1
def split(image_dataset,image_num,label_dataset,ratio):
    train_num=int(image_num*ratio)
    indice=np.random.permutation(image_num)
    image_dataset=image_dataset[indice]
    label_dataset=label_dataset[indice]
    train_image_dataset=image_dataset[:train_num]
    train_label_dataset=label_dataset[:train_num]
    validation_image_dataset=image_dataset[train_num:]
    validation_label_dataset=label_dataset[train_num:]
    return train_num,train_image_dataset,train_label_dataset,validation_image_dataset,validation_label_dataset

#further split training data into batches
#image_batch=[[image][image]]
#batch=[(image_batch,label_batch),(image_batch,label_batch)]
def split_training_data(train_image_dataset,train_label_dataset,batch_size,train_num,batch_num):
    image_batch=[]
    label_batch=[]
    for i in range(batch_num):
        start=i*batch_size
        end=min(train_num,start+batch_size)
        image_batch.append(train_image_dataset[start:end])
        label_batch.append(train_label_dataset[start:end])
    return np.array(image_batch),np.array(label_batch)


###forward
##convolution
#size for the length of the convolution kernel size; iamge for the input; batch_size for the number of images in each input
#padding_len for the width of the padding, padding_val for the number inserted, conv_size for the output size for convolution
#this creates a vector which demonstrates the result
def convolution(size,image_batch,padding_len,padding_val,batch_size,image_row):
    padding_vec=((0,0),(padding_len,padding_len),(padding_len,padding_len))
    image_batch=np.pad(image_batch,padding_vec,mode='constant',constant_values=padding_val)
    conv=np.random.rand(size,size)*2-1 #[-1,1]
    conv_size=image_row+padding_len*2-size+1
    conv_result=np.zeros((batch_size,conv_size,conv_size))
    for i in range(batch_size):
        for j in range(conv_size):
            for k in range(conv_size):
                image_temp=image_batch[i,j:j+size,k:k+size]
                conv_result[i,j,k]=np.sum(image_temp*conv)
    return conv_result

##ReLU
def ReLU(conv_result):
    return np.clip(conv_result,0,None)

##pooling
#convert conv_result[size,size] into pool_result[size/2,size/2] as default
#functions include:
    #max pool vs mean pool
def pooling(conv_result,pool_function):
    batch_size=conv_result.shape[0]
    pool_size=(conv_result.shape[1]+1)//2
    pool_result=np.zeros((batch_size,pool_size,pool_size))

    for i in range(batch_size):
        for j in range(pool_size):
            for k in range(pool_size):
                pool_temp=conv_result[i,j*2:j*2+2,k*2:k*2+2]
                if pool_function=="max pool":
                    pool_result[i,j,k]=np.max(pool_temp)
                if pool_function=="mean pool":
                    pool_result[i,j,k]=np.mean(pool_temp)

    return pool_result


#flatten
#image_batch: shape[batch,row,col]
#flatten_result: shape[batch,vector]
def flatten(image_batch):
    batch_size=image_batch.shape[0]   
    flatten_result=image_batch.reshape(batch_size,-1)
    return flatten_result

### 1-d vector part, based on the sinx part
#initialize the number of elements in nn
def initialize_nn_count(image_row,latent_layer_num,neuron_list):
    count=np.zeros(latent_layer_num+2,dtype=int)
    for i in range(latent_layer_num):
        count[i+1]=neuron_list[i]
    count[0]=image_row*image_row
    count[latent_layer_num+1]=10
    return count

#initialize w,b with normal distribution
# for count nn
# nn[i]=[weight,bias]
# i for layer closer to input,j for layer closer to output
# weight[i][j]=the weight of Oi to Oj
# bias[j]=the bias of Oj
def initialize_nn_parameters(count):
    layer=count.size
    nn=[]
    for i in range(layer-1):
        std=np.sqrt(2.0/(count[i+1]+count[i]))
        weight=np.random.normal(0,std,(count[i+1],count[i]))
        bias = np.random.normal(0, 0.1, (count[i + 1],))
        nn.append((weight,bias))
    return nn

#define tanh function
def tanh(x):
    return np.tanh(x)

#define tanh derivative function
def tanh_derivative(x):
    return 1-np.tanh(x)**2

#define softmax
def softmax(x):
    e_x=np.exp(x-np.max(x,axis=1,keepdims=True))
    return e_x/e_x.sum(axis=1,keepdims=True)

##forward_propagation: for oi in layer x, x starts from 0
#this calculates the value of each neuron
def forward_propagation(nn_parameters,flatten_result,count):
    activation=[flatten_result]
    layer=count.size
    for i in range(layer-1):
        weight,bias=nn_parameters[i]
        act=activation[-1]@weight.T+bias
        if i!=layer-2:                                              # maybe dont need?
            act=tanh(act)
        activation.append(act)

    #softmax
    activation[-1]=softmax(activation[-1])

    return activation


##backward_propagation:
#delta parameter matrix: calculates the delta
#initialize delta_matrix:
#each layer of delta_matrix contains error, w, b, which simplifies the expression of cal_delta_matrix
def initialize_delta_matrix(count,batch_size):
    delta_matrix=[]
    delta_softmax_layer={
        "error":np.zeros((batch_size,count[-1]))
    }
    delta_matrix.append(delta_softmax_layer)
    for i in range(count.size-1):
        delta_matrix_layer={
        "error":np.zeros((batch_size,count[i+1])),
        "w":np.zeros((count[i+1],count[i])),
        "b":np.zeros((count[i+1],))
        }
        delta_matrix.append(delta_matrix_layer)
    return delta_matrix

#calculate delta_matrix:
#label_batch: shape[batch,vector]
def cal_delta_matrix(label_batch,activation,count,learning_rate,nn_parameters):
    delta_matrix=initialize_delta_matrix(count,label_batch.shape[0])
    batch_size=label_batch.shape[0]

    #for softmax layer:
    delta_matrix[-1]["error"]=label_batch-activation[-1]

    #for other layers:
    for layer in range(count.size-2,-1,-1):
        weight=nn_parameters[layer][0]
        delta_next=delta_matrix[layer+1]["error"]
        act=activation[layer]

        delta_matrix[layer]["error"]=(delta_next@weight)*(tanh_derivative(act) if layer>0 else 1)
        delta_matrix[layer]["w"]=learning_rate*(act.T@delta_next)/batch_size
        delta_matrix[layer]["b"]=learning_rate*delta_next.sum(axis=0)/batch_size

    return delta_matrix
 
#train nn in a batch
#epoch_num: how many epochs altogether; epoch: the epoch_th epoch
#loss_mini_batch shows the change of loss with each mini batch; loss_train shows the change of loss with each examine_num; loss_validation shows the change of loss on validation dataset with each examine_num;list tracks the change of epoch
def train_mini_batch(conv_size,nn_parameters,image_batch,padding_len,padding_val,batch_size,image_row,pool_function,count,label_batch,learning_rate):
    conv_result=convolution(conv_size,image_batch,padding_len,padding_val,batch_size,image_row)
    conv_result=ReLU(conv_result)
    pool_result=pooling(conv_result,pool_function)
    flatten_result=flatten(pool_result)
    activation=forward_propagation(nn_parameters,flatten_result,count)
    delta_matrix=cal_delta_matrix(label_batch,activation,count,learning_rate,nn_parameters)
    for layer in range(count.size-1):
        weight,bias=nn_parameters[layer]
        weight+=delta_matrix[layer]["w"].T
        bias+=delta_matrix[layer]["b"]
        nn_parameters[layer]=(weight,bias)
    loss_temp=np.mean(np.abs(label_batch-activation[-1]))
    return loss_temp
    


def train_nn(batch_num,conv_size,nn_parameters,image_all,count,padding_len,padding_val,batch_size,image_row,pool_function,label_all,learning_rate,epoch_num,examine_num,validation_image_dataset,validation_label_dataset):
    loss_mini_batch=[]
    loss_train=[]
    loss_validation=[]
    tlist=[]
    print(f"examine_num: {examine_num}")                                    #further shuffle the batch?
    for epoch in range(epoch_num):
        print(f"epoch: {epoch}")
        loss_temp=0
        for image_batch_num in range(batch_num):
            loss_temp+=train_mini_batch(conv_size,nn_parameters,image_all[image_batch_num],padding_len,padding_val,batch_size,image_row,pool_function,count,label_all[image_batch_num],learning_rate)
        loss_temp/=batch_num
        loss_mini_batch.append(loss_temp)
        tlist.append(epoch)
        loss_train.append(loss_temp)

        if epoch%examine_num==0 or epoch==epoch_num-1:
            val_flatten=flatten(validation_image_dataset)
            val_activation=forward_propagation(nn_parameters,val_flatten,count)
            val_loss_temp=np.mean(np.abs(validation_label_dataset-val_activation[-1]))
            
            loss_validation.append(val_loss_temp)
            print(f"epoch {epoch}: train loss = {loss_temp:.6f}, val loss = {val_loss_temp:.6f}")
        else:
            loss_validation.append(np.nan)
    return loss_mini_batch,loss_train,loss_validation,tlist

#calculate accuracy
def cal_accuracy(nn_parameters,validation_image_dataset,validation_label_dataset,count):
    predictions=[]
    ground_truth=[]
    for image,label in zip(validation_image_dataset,validation_label_dataset):
        flatten_result=flatten(np.array([image]))
        prediction=np.argmax(forward_propagation(nn_parameters,flatten_result,count)[-1])
        predictions.append(prediction)
        ground_truth.append(np.argmax(label))
    return accuracy_score(ground_truth,predictions)

if __name__ == "__main__":
    #initializing:
    image_dataset,image_num,image_row=read_image_dataset(r"train-images.idx3-ubyte")
    label_dataset=read_label_dataset(r"train-labels.idx1-ubyte")

    #interaction:
    ratio=float(input("你想要将数据集按多少的比例进行切分："))
    conv_size=int(input("你想要的卷积核大小："))
    padding_len=int(input("你想要的padding宽度："))
    padding_val=int(input("你想要padding被填充的值是多少："))
    batch_size=int(input("你想要的batch size:")) #num of images in a batch
    latent_layer_num=int(input("你想要的隐藏层数量："))
    print(f"你每一层需要几个神经元:")
    if latent_layer_num==1:
        neuron_list=[int(input())];
    else:
        neurons=input().split();
        neuron_list=[int(neuron) for neuron in neurons];
    learning_rate=float(input("你想要的学习率："));
    epoch_num=int(input("你想要训练多少轮："));
    examine_num=int(input("你想要多少epoch检查一次："))

    #initialize
    train_num,train_image_dataset,train_label_dataset,validation_image_dataset,validation_label_dataset=split(image_dataset,image_num,label_dataset,ratio)
    batch_num=(train_num+batch_size-1)//batch_size #一共有多少个batch
    image_all,label_all=split_training_data(train_image_dataset,train_label_dataset,batch_size,train_num,batch_num) #[[image_batch][image_batch]] image=[[image][image]]
    count=initialize_nn_count(image_row,latent_layer_num,neuron_list)
    nn_parameters=initialize_nn_parameters(count)

    #train
    loss_mini_batch,loss_train,loss_validation,tlist=train_nn(batch_num,nn_parameters,image_all,count,label_all,learning_rate,epoch_num,examine_num,validation_image_dataset,validation_label_dataset)

    #visualizing
    plt.plot(tlist,loss_train,label="train loss")
    plt.plot(tlist,loss_validation,'o-',label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("loss over epoch")
    plt.show()

    #cal_accuracy
    accuracy=cal_accuracy(nn_parameters,validation_image_dataset,validation_label_dataset,count)
    print(f"正确率：{accuracy}")