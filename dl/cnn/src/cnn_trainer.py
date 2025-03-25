import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
import numpy as np
from tqdm import tqdm

DATA_FOLDER="G:/my drive/fau/egn4952c_spring_2025/data/spec_subset"
LABEL_INDEX={"Seizure":0,"LRDA":1,"GRDA":2,"LPD":3,"GPD":4,"Other":5}
BATCH_SIZE=1
LEARNING_RATE=1e-3
NUM_EPOCHS=1
VAL_SPLIT=0.2
SEED=42
torch.manual_seed(SEED)
np.random.seed(SEED)

def train(model,dataloader,criterion,optimizer,device):
  model.train()
  running_loss=0.0
  correct=0
  total=0
  for batch in tqdm(dataloader,desc="train"):
    spectrograms,labels=batch
    spectrograms=spectrograms.to(device)
    labels=labels.to(device)
    outputs=model(spectrograms)
    loss=criterion(outputs,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss+=loss.item()*spectrograms.size(0)
    _,pred=torch.max(outputs,1)
    correct+=(pred==labels).sum().item()
    total+=labels.size(0)
  return running_loss/total,correct/total

def validate(model,dataloader,criterion,device):
  model.eval()
  running_loss=0.0
  correct=0
  total=0
  with torch.no_grad():
    for batch in tqdm(dataloader,desc="val"):
      spectrograms,labels=batch
      spectrograms=spectrograms.to(device)
      labels=labels.to(device)
      outputs=model(spectrograms)
      loss=criterion(outputs,labels)
      running_loss+=loss.item()*spectrograms.size(0)
      _,pred=torch.max(outputs,1)
      correct+=(pred==labels).sum().item()
      total+=labels.size(0)
  return running_loss/total,correct/total

def main():
  device=torch.device("cuda"if torch.cuda.is_available()else"cpu")
  dataset=SpecDataset(data_folder=DATA_FOLDER,label_index=LABEL_INDEX)
  val_size=int(len(dataset)*VAL_SPLIT)
  train_size=len(dataset)-val_size
  train_dataset,val_dataset=random_split(dataset,[train_size,val_size])
  train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
  val_loader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=2)
  model=SpectrogramCNN(num_classes=len(LABEL_INDEX)).to(device)
  criterion=nn.CrossEntropyLoss()
  optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE)
  best_val_acc=0.0
  model_path="spectrogram_cnn.pth"
  for epoch in range(NUM_EPOCHS):
    train_loss,train_acc=train(model,train_loader,criterion,optimizer,device)
    val_loss,val_acc=validate(model,val_loader,criterion,device)
    print(f"Epoch[{epoch+1}/{NUM_EPOCHS}]TrainLoss:{train_loss:.4f},TrainAcc:{train_acc:.4f}|ValLoss:{val_loss:.4f},ValAcc:{val_acc:.4f}")
    if val_acc>best_val_acc:
      best_val_acc=val_acc
      torch.save(model.state_dict(),model_path)
      print(f"New best model saved at epoch {epoch+1} with val_acc={val_acc:.4f}")
  print("Training complete!")
  print(f"Best validation accuracy:{best_val_acc:.4f}")
  print(f"Best model is saved to:{model_path}")

if __name__=="__main__":
  main()
