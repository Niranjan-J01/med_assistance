import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch as ts
import torch.nn as nn
from sklearn.metrics import confusion_matrix , classification_report

device = ts.device("cuda")

from torchvision.transforms import transforms

trans = transforms.Compose([
    transforms.ToTensor(),# image to tensor -> scale(0,1)
    transforms.Normalize((0.5 , 0.5 , 0.5 ) , (0.5 , 0.5 , 0.5 )) # set of values for normalizing( -1 , 1)
]) 

from torchvision.datasets import CIFAR10

traindata = CIFAR10(root="./base task" , train=True , download= True , transform=trans)
testdata = CIFAR10(root="./base task" , train=False , download= True , transform=trans)

from torch.utils.data import DataLoader

train_x = DataLoader( traindata, batch_size =  64 , shuffle = True)
test_x = DataLoader(testdata , batch_size=64 )

# Residual block 

class resblock(nn.Module):

    def __init__(self , in_channel , out_channel , stride):
        
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel , out_channels=out_channel , kernel_size=3 , padding=1 , stride=stride)

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channel , out_channels=out_channel , kernel_size=3 , padding=1 )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

        if stride != 1 or in_channel != out_channel :

            self.shortcut = nn.Sequential(

                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=stride
                ),
                nn.BatchNorm2d(out_channel)

            )
        else :

            self.shortcut = nn.Identity()
    
    def forward(self , x):

        out = self.relu(self.bn1(self.conv1(x)))  # Conv1 -> bn > relu
        out = self.bn2(self.conv2(out)) # conv2 -> bn 
        out = out + self.shortcut(x) # adding skip connectionn 

        out = self.relu(out) # final relu

        return out 
    
class cumtomresnet(nn.Module):

    def __init__(self, f_out = 10):
        super().__init__()

        self.start = nn.Sequential(
            nn.Conv2d(
                3 , 64 ,
                kernel_size=3 ,
                stride=1 , 
                padding= 1
            ),
            nn.BatchNorm2d(64) , 
            nn.ReLU()
        )

        self.stage_1 = nn.Sequential(
            resblock(64 , 64 , stride = 1),
            resblock(64 , 64 , stride = 1)
        )

        self.stage_2 = nn.Sequential(
            resblock(64 , 128 , stride = 2 ),
            resblock(128 , 128 , stride= 1)
        )

        self.stage_3 = nn.Sequential(
            resblock(128 , 256 , stride = 2),
            resblock(256 , 256 , stride = 1)
        )

        self.relu = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.linear = nn.Linear(256 , f_out)

    def forward(self , x ):

        x = self.start(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.pool(x)
        x = x.view(x.size(0) , -1)
        x = self.linear(x)

        return x 

model = cumtomresnet().to(device)
critirian = nn.CrossEntropyLoss()
optim = ts.optim.Adam(model.parameters())

epochs = 25
best_model = float("inf")
train_loss = []
val_loss  = []

for epoch in range(epochs):

    model.train()
    running_losses = 0.00

    for image , label in train_x:
        image = image.to(device)
        label = label.to(device)

        optim.zero_grad()
        outputs = model(image)
        loss = critirian(outputs , label)
        loss.backward()
        optim.step()

        running_losses += loss.item()
    
    epoch_losses = running_losses/len(train_x)
    train_loss.append(epoch_losses)

    model.eval()
    total =0 
    correct = 0 

    all_preds, all_labels = [], []
    with ts.no_grad():

        running_val_losses = 0.00
        for image , label in test_x:

            image , label = image.to(device), label.to(device)

            output = model(image)
            loss  = critirian(output , label)
            running_val_losses += loss

            _ , pred = ts.max(output , 1)
            correct += (label == pred).sum().item()
            total += label.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

            #print(classification_report(all_labels, all_preds, target_names=traindata.classes))

            accuracy = 100 * correct/total
            epoch_val_loss = running_val_losses/len(test_x)
            val_loss.append(epoch_val_loss)
        
        print(f"""epoch {1+epoch} : train_loss = {epoch_losses} & test_loss = {epoch_val_loss}
               $$ accuracy : {accuracy} """)

import pandas as pd 
import matplotlib as plt

loss_df = pd.DataFrame({
    "train_loss" : train_loss,
    "val_loss" : val_loss
})

plt.plot(loss_df["Training Loss"], label = "Training Loss")
plt.plot(loss_df["Validation Loss"], label = "Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Losses")

plt.legend()
plt.show()

        
        







        
