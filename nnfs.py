import  numpy as np
from numpy import random
import math
np.random.seed(0)
n=3
batch=3
n2=3
input=random.rand(batch,n) #unique sensors input to map information tell us from those
w=random.rand(n, n)
b=random.rand(n)
w2=random.rand(n, n)
b2=random.rand(n)
out=np.dot(input,np.array(w).T)+b
out2=np.dot(out,np.array(w2).T)+b2
print(out2)



class Layer_Dense:
    def __init__(self,n_input,n_neu):
        self.w= np.random.randn(n_input,n_neu)
        self.b= np.zeros((1,n_neu))
    def forward(self,input):
        self.output=np.dot(input,self.w)+self.b
class Activation_ReLU():
    def forward(self,input_11):
        self.output=np.maximum(0,input_11)
n=3  #sensor inputs
batch=30 #number of input set used per one computation
n_out=2 # output dimension
input=random.rand(batch,n) #unique sensors input to map information tell us from those

layer_1=Layer_Dense(n,batch)
layer_2=Layer_Dense(batch,n_out)
layer_1.forward(input)
print(layer_1.output)
layer_2.forward(layer_1.output)
print(layer_2.output)
activation1=Activation_ReLU()
activation1.forward(layer_2.output)
print(activation1.output)


class Activation_Softmax():
    def forward(self,input):
        exp_values=np.exp(input-np.max(input, axis=1,keepdims=True))
        prob= exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output=prob
        
layer_1=Layer_Dense(n,batch)
layer_2=Layer_Dense(batch,n_out)
activation1=Activation_ReLU()
activation2=Activation_Softmax()
layer_1.forward(input)
activation1.forward(layer_1.output)
print(activation1.output)
layer_2.forward(activation1.output)
activation2.forward(layer_2.output)
print(activation2.output)


out_d=np.array([1 ,0, 0])
soft_max_out=np.array([0.7, 0.1,0.2])
loss=0
dim=3
ind=0
for i in range(dim):
    loss+=-(math.log(soft_max_out[ind])*out_d[ind])
    ind+=1
    
print(loss)

class Loss:
    def calculate(self,output,y):
        salmple_losses = self.forward(output,y)
        data_loss= np.mean(salmple_losses)
        return data_loss

class Loss_cross_entropy(Loss):
    def forward(self,y_pred,y_true):
        sample=len(y_pred)
        y_pre_clip=np.clip(y_pred, 1e-7,1-1e-7)
        
        if len(y_true.shape) == 1:
            correct_confi= y_pre_clip[range(sample),y_true]
        else:
            correct_confi= np.sum(y_pre_clip*y_true, axis=1)
        neg_log_lik_hood=-np.log(correct_confi)
        return neg_log_lik_hood
    
out_d=np.array([1 ,0, 0])
soft_max_out=np.array([[0.7, 0.1,0.2],[0.5,0.1,0.0],[0.5,0.2,0.1]])
Loss_function= Loss_cross_entropy()
loss =Loss_function.calculate(soft_max_out,out_d)

print(loss)

predictions= np.argmax(soft_max_out, axis=1)
accuracy=np.mean(predictions==out_d)
print(accuracy)



n=3  #sensor inputs
batch=30 #number of input set used per one computation
n_out=2 # output dimension
input=random.rand(batch,n) #unique sensors input to map information tell us from those
out_d=random.rand(n_out,1) 
y=out_d

class Layer_Dense1:
    def __init__(self,w,b):
        self.w= w #  batch by n
        self.b= b # batch by 1
    def forward(self,input):
        self.output=np.dot(input,self.w)+self.b


# Objective function (minimize this)
def objective(x1,x2,b1,b2,y,in_1,batch):
    Loss_function= Loss_cross_entropy()
    layer_1=Layer_Dense1(x1,b1)
    layer_2=Layer_Dense1(x2,b2)
    activation1=Activation_ReLU()
    activation2=Activation_Softmax()
    
    layer_1.forward(np.array(in_1))
    activation1.forward(layer_1.output)
   # print(activation1.output)
    layer_2.forward(activation1.output)
    activation2.forward(layer_2.output)
    soft_max_out=activation2.output
   # loss =Loss_function.calculate(soft_max_out,y)
    predictions= np.max(soft_max_out, axis=1)
    accuracy=np.mean(predictions-y)    

    print(predictions)
    
    return accuracy  # Simple sphere function

m=8
input=random.rand(batch,n) #unique sensors input to map information tell us from those
out_d=random.rand(n_out,1) 
y=out_d
# PSO Parameters
num_particles = 300
dimensions = n
iterations = 100
w = 0.5       # inertia
c1 = 1.5      # cognitive
c2 = 1.5      # social


# Initialize particle positions and velocities
w1 = np.random.randn(n,m)
w2 = np.random.randn(m,n_out)
b1 = np.random.randn(m)
b2 = np.random.randn(n_out)
vel = []
w1_best = w1.copy()
w2_best = w2.copy()
b1_best = b1.copy()
b2_best = b2.copy()

w1_global_best = w1.copy()
w2_global_best = w2.copy()
b1_global_best = b1.copy()
b2_global_best = b2.copy()
def w_val(w1,w1_best,w1_global_best):
    w = 0.5       # inertia
    c1 = 1.5      # cognitive
    c2 = 1.5      # social
    
    r1, r2 = np.random.rand(), np.random.rand()
    w1 = (w * w1+ c1 * r1 * (w1_best - w1)+ c2 * r2 * (w1_global_best - w1))
           
    return w1
# Main PSO loop
best_score=10000000
global_best=1000000
for t in range(iterations):
    for i in range(num_particles):
        
            x1 = w_val(w1,w1_best,w1_global_best)
            x2 = w_val(w2,w2_best,w2_global_best)
            b1 = w_val(b1,b1_best,b1_global_best)
            b2 = w_val(b2,b2_best,b2_global_best)
            
        
            score =np.abs(objective(x1,x2,b1,b2,out_d,input,batch))
            print(score)
            if score < best_score:
                best_score = score
                w1_best = x1.copy()
                w2_best = x2.copy()
                b1_best = b1.copy()
                b2_best = b2.copy()
                
    if best_score <global_best:
                    global_best = best_score
                   
                    w1_global_best = w1_best 
                    w2_global_best = w2_best 
                    b1_global_best = b1_best
                    b2_global_best = b2_best

print("Best solution:", global_best)
print("Best score:", w1_global_best,w2_global_best,b1_global_best,b2_global_best)
