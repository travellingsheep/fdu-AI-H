import struct
import numpy as np
from sklearn.metrics import accuracy_score

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

class FC:
    def __init__(self,image_row,latent_layer_num,neuron_list,learning_rate):
        self.image_row=image_row
        self.latent_layer_num=latent_layer_num
        self.neuron_list=neuron_list
        self.count=self.initialize_nn_count()
        self.nn=self.initialize_nn_parameters()
        self.learning_rate=learning_rate

    def initialize_nn_count(self):
        image_row=self.image_row
        latent_layer_num=self.latent_layer_num
        neuron_list=self.neuron_list
        count=np.zeros(latent_layer_num+2,dtype=int)
        for i in range(latent_layer_num):
            count[i+1]=neuron_list[i]
        count[0]=image_row*image_row
        count[latent_layer_num+1]=10
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

    @staticmethod
    def ReLU(conv_result):
        return np.clip(conv_result,0,None)

    @staticmethod
    def ReLU_derivative(conv_result):
        return (conv_result > 0).astype(float)

    @staticmethod
    def softmax(x):
        e_x=np.exp(x-np.max(x,axis=1,keepdims=True))
        return e_x/e_x.sum(axis=1,keepdims=True)

    @staticmethod
    def flatten(image_batch):
        batch_size=image_batch.shape[0]   
        flatten_result=image_batch.reshape(batch_size,-1)
        return flatten_result

    def forward_propagation(self,flatten_result):
        nn_parameters=self.nn
        count=self.count
        activation=[flatten_result]
        layer=count.size
        for i in range(layer-1):
            weight,bias=nn_parameters[i]
            act=activation[-1]@weight.T+bias
            if i!=layer-2:
                act=self.ReLU(act)
            activation.append(act)

        activation[-1]=self.softmax(activation[-1])

        return activation

    def initialize_delta_matrix(self,count,batch_size):
        count=self.count
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

    def cal_delta_matrix(self,label_batch,flatten_result):
        count=self.count
        activation=self.forward_propagation(flatten_result)
        learning_rate=self.learning_rate
        nn_parameters=self.nn
        delta_matrix=self.initialize_delta_matrix(count,label_batch.shape[0])
        batch_size=label_batch.shape[0]

        delta_matrix[-1]["error"]=label_batch-activation[-1]

        for layer in range(count.size-2,-1,-1):
            weight=nn_parameters[layer][0]
            delta_next=delta_matrix[layer+1]["error"]
            act=activation[layer]

            delta_matrix[layer]["error"]=(delta_next@weight)*(self.ReLU_derivative(act) if layer>0 else 1)
            delta_matrix[layer]["w"]=learning_rate*(act.T@delta_next)/batch_size
            delta_matrix[layer]["b"]=learning_rate*delta_next.sum(axis=0)/batch_size

        return delta_matrix

    def train_mini_batch(self,image_batch,label_batch):
        nn_parameters=self.nn
        count=self.count
        flatten_result=self.flatten(image_batch)
        activation=self.forward_propagation(flatten_result)
        delta_matrix=self.cal_delta_matrix(label_batch,flatten_result)
        for layer in range(count.size-1):
            weight,bias=nn_parameters[layer]
            weight+=delta_matrix[layer]["w"].T
            bias+=delta_matrix[layer]["b"]
            nn_parameters[layer]=(weight,bias)
        exp_p=np.exp(activation[-1])
        sum_exp_p=np.sum(exp_p,axis=1,keepdims=True)
        exp_p/=sum_exp_p
        exp_p=np.log(exp_p)
        exp_temp=-np.sum(exp_p*label_batch,axis=1)
        loss_temp=np.mean(exp_temp)
        return loss_temp

    def train_nn(self,batch_num,image_all,label_all,epoch_num,examine_num,validation_image_dataset,validation_label_dataset):
        loss_mini_batch=[]
        loss_train=[]
        loss_validation=[]
        tlist=[]
        print(f"examine_num: {examine_num}")
        for epoch in range(epoch_num):
            print(f"epoch: {epoch}")
            loss_temp=0
            for image_batch_num in range(batch_num):
                loss_temp+=self.train_mini_batch(image_all[image_batch_num],label_all[image_batch_num])
            loss_temp/=batch_num
            loss_mini_batch.append(loss_temp)
            tlist.append(epoch)
            loss_train.append(loss_temp)

            if epoch%examine_num==0 or epoch==epoch_num-1:
                val_flatten=self.flatten(validation_image_dataset)
                val_activation=self.forward_propagation(val_flatten)
                val_exp_p=np.exp(val_activation[-1])
                val_sum_exp_p=np.sum(val_exp_p,axis=1,keepdims=True)
                val_exp_p/=val_sum_exp_p
                val_exp_p=np.log(val_exp_p)
                val_exp_temp=-np.sum(val_exp_p*validation_label_dataset,axis=1)
                val_loss_temp=np.mean(val_exp_temp)
                loss_validation.append(val_loss_temp)
                print(f"epoch {epoch}: train loss = {loss_temp:.6f}, val loss = {val_loss_temp:.6f}")
            else:
                loss_validation.append(np.nan)
        return loss_mini_batch,loss_train,loss_validation,tlist

    #calculate accuracy
    def cal_accuracy(self,validation_image_dataset,validation_label_dataset):
        predictions=[]
        ground_truth=[]
        for image,label in zip(validation_image_dataset,validation_label_dataset):
            flatten_result=self.flatten(np.array([image]))
            prediction=np.argmax(self.forward_propagation(flatten_result)[-1])
            predictions.append(prediction)
            ground_truth.append(np.argmax(label))
        return accuracy_score(ground_truth,predictions)
    
    def interview(self,eval_datafile_path):
        eval_datafile_path_image=eval_datafile_path[0]
        eval_datafile_path_label=eval_datafile_path[1]
        image_dataset,_,_=read_image_dataset(eval_datafile_path_image)
        label_dataset=read_label_dataset(eval_datafile_path_label)
        predictions=[]
        ground_truth=[]
        for image,label in zip(image_dataset,label_dataset):
            flatten_result=self.flatten(np.array([image]))
            prediction=np.argmax(self.forward_propagation(flatten_result)[-1])
            predictions.append(prediction)
            ground_truth.append(np.argmax(label))
        return 100*accuracy_score(ground_truth,predictions)