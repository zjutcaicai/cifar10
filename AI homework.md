# Homework

### 要求

1、Creating your own github account.

2、Implementing your own deep neural network (in Pytorch, PaddlePaddle…).

3、Training it on CIFAR10.

4、Tuning a hyper-parameter and analyzing its effects on performance.

5、Writing a README.md to report your findings.

### 超参数调节和分析

主要对训练的epoch、batch_size和learning rate进行调节，找到最优的准确率。epoch增加可以更好的收敛，batch_size主要是并行处理，增大可以更好找到梯度方向。learning rate可以调节收敛的快慢。如下表，第六次，epoch在80，batch_size 32,learning rate 1e-2是目前最好的结果，相较于前面，基本都是增大对应参数

| 次数 | epoch | batch_size | 学习率 | 正则项系数 | accuracy |
| ---- | ----- | ---------- | ------ | ---------- | -------- |
| 1    | 10    | 8          | 1e-3   | 0          | 60.5%    |
| 2    | 20    | 8          | 1e-2   | 0          | 67.6%    |
| 3    | 20    | 16         | 1e-2   | 0          | 70.9%    |
| 4    | 40    | 16         | 1e-2   | 0          | 83.1%    |
| 5    | 50    | 32         | 1e-2   | 0          | 89.5%    |
| 6    | 80    | 32         | 1e-2   | 0          | 90.1%    |



### 数据集准备和处理

```python
import os
import torch
import torchvision
import torchvision.transforms as transforms

classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')#10个类

# 定义对数据的预处理
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform )
trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)
testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=False,transform=transform )
testloader=torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)


```



### 网络

模型参考较多博客，进行魔改。

```python

import torch.nn as nn
import torch.nn.functional as F
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 56, 1)
        self.conv1_bn = nn.BatchNorm2d(56)
        self.conv2 = nn.Conv2d(56, 84, 2)
        self.conv2_bn = nn.BatchNorm2d(84)
        self.conv3 = nn.Conv2d(84, 128, 2)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 2)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 2)
        self.conv5_bn = nn.BatchNorm2d(512)
        self.drop = nn.Dropout2d(p=0.25)
        
        self.fc1 = nn.Linear(4608, 2000)          
        self.fc2 = nn.Linear(2000, 10)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x))) #Conv -> BN -> ReLu
        x = F.relu(self.conv2_bn(self.conv2(x))) #Conv -> BN -> ReLu
        x = F.max_pool2d(F.relu(self.conv3_bn(self.conv3(x))),2) #Conv -> BN -> ReLu -> Max Pooling
        x = F.max_pool2d(F.relu(self.conv4_bn(self.conv4(x))),2) #Conv -> BN -> ReLu -> Max Pooling 
        x = F.max_pool2d(F.relu(self.conv5_bn(self.conv5(x))),2) #Conv -> BN -> ReLu -> Max Pooling

        x = self.drop(x)
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

```



### 训练和测试

```python

#初始化参数
for m in net.modules():
    if isinstance(m,nn.Conv2d):
        nn.init.normal_(m.weight)
        nn.init.xavier_normal_(m.weight)
        nn.init.kaiming_normal_(m.weight)  
        nn.init.constant_(m.bias,0)
    elif isinstance(m,nn.Linear):
        nn.init.normal_(m.weight)  
        

import torch.optim as optim
criterion=nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

if __name__ == '__main__':
     for epoch in range(20):
         running_loss=0.0
         for i,data in enumerate(trainloader,0):
             inputs,labels=data
             inputs,labels=inputs.to(device),labels.to(device)
        
             optimizer.zero_grad()
        
             output=net(inputs)
             loss=criterion(output,labels)
             loss.backward()
             optimizer.step()

             running_loss+=loss.item()
             if i % 2000 == 1999: 
                 print('[%d,%5d] loss:%.3f' %(epoch+1, i+1, running_loss/2000))
                 running_loss = 0.0
     print('finish training')

    
correct=0
total=0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        output = net(inputs)
        _,predicted=torch.max(output.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
print('accuracy:%d %%'%(100*correct/total))


class_correct=list(0.for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        output = net(inputs)
        _,predicted=torch.max(output,1) 
        c=(predicted==labels).squeeze() 
        for i in range(4):  
            label=labels[i]  
            class_correct[label]+=c[i].item()
            class_total[label] +=1
for i in range(10):
     print('accuracy of %5s:%2d %%'%(classes[i],100*class_correct[i]/class_total[i]))
```

