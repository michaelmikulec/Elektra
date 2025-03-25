import os, torch, torch.nn as nn, torch.optim as optim, numpy as np
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm
from eeg_dataset import EEGDataset
from eeg_transformer import EEGTransformer

torch.set_num_threads(22);torch.set_num_interop_threads(1);
def train(model,loader,criterion,optimizer,device):
  float:running_loss=0.0;int:correct=0;int:total=0;model.train();
  for batch in tqdm(loader,desc="train"):
    data,labels=batch; data=data.to(device,non_blocking=True); labels=labels.to(device,non_blocking=True); outputs=model(data); loss=criterion(outputs,labels);
    optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); running_loss+=loss.item()*data.size(0);
    _,pred=torch.max(outputs,1); correct+=(pred==labels).sum().item(); total+=labels.size(0); return running_loss/total,correct/total;
def validate(model,loader,criterion,device):
  float:running_loss=0.0;int:correct=0;int:total=0;model.eval();
  with torch.no_grad():
    for batch in tqdm(loader,desc="val"): 
      data,labels=batch; data=data.to(device,non_blocking=True); labels=labels.to(device,non_blocking=True); outputs=model(data);
      loss=criterion(outputs,labels); running_loss+=loss.item()*data.size(0); _,pred=torch.max(outputs,1); correct+=(pred==labels).sum().item(); 
      total+=labels.size(0);
  return running_loss/total,correct/total;
def main():
  MODELS_FOLDER="G:/My Drive/fau/egn4952c_spring_2025/deep_learning/dev/models"; MODEL="transfm.pth";
  DATA_FOLDER="G:/My Drive/fau/egn4952c_spring_2025/data/1200eeg"; LABEL_INDEX={"Seizure":0,"LRDA":1,"GRDA":2,"LPD":3,"GPD":4,"Other":5};
  INPUT_DIM=20;MODEL_DIM=128;NUM_HEADS=4; NUM_LAYERS=2;DIM_FEEDFORWARD=256;DROPOUT=0.1; NUM_CLASSES=len(LABEL_INDEX);BATCH_SIZE=1;
  LEARNING_RATE=1e-6;NUM_WORKERS=16; NUM_EPOCHS=1;VAL_SPLIT=0.2;SEED=42; torch.manual_seed(SEED);np.random.seed(SEED);
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); dataset=EEGDataset(data_folder=DATA_FOLDER,label_index=LABEL_INDEX);
  val_size=int(len(dataset)*VAL_SPLIT); train_size=len(dataset)-val_size; train_dataset,val_dataset=random_split(dataset,[train_size,val_size]);
  train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS, pin_memory=True,persistent_workers=True,prefetch_factor=4);
  val_loader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS, pin_memory=True,persistent_workers=True,prefetch_factor=4);
  transfm=EEGTransformer(input_dim=INPUT_DIM,model_dim=MODEL_DIM,num_heads=NUM_HEADS, num_layers=NUM_LAYERS,dim_feedforward=DIM_FEEDFORWARD,dropout=DROPOUT, num_classes=NUM_CLASSES);
  if torch.cuda.device_count()>1:transfm=nn.DataParallel(transfm);print("DataParallel in use."); transfm.to(device);checkpoint_path=os.path.join(MODELS_FOLDER,MODEL);
  if os.path.exists(checkpoint_path):
    print(f"Loading existing model from {checkpoint_path}..."); state_dict=torch.load(checkpoint_path,map_location=device);
    if isinstance(transfm,nn.DataParallel):transfm.module.load_state_dict(state_dict);
    else:transfm.load_state_dict(state_dict);
  else:print("No existing model to load.");
  criterion=nn.CrossEntropyLoss(); optimizer=optim.Adam(transfm.parameters(),lr=LEARNING_RATE);
  for epoch in range(NUM_EPOCHS):
    train_loss,train_acc=train(transfm,train_loader,criterion,optimizer,device); val_loss,val_acc=validate(transfm,val_loader,criterion,device);
    print(f"Epoch[{epoch+1}/{NUM_EPOCHS}]\nTrainLoss:{train_loss:.4f},TrainAcc:{train_acc:.4f} | ValLoss:{val_loss:.4f},ValAcc:{val_acc:.4f}");
    print("Saving final model to transfm.pth...");
    if isinstance(transfm,nn.DataParallel):torch.save(transfm.module.state_dict(),checkpoint_path)
    else:torch.save(transfm.state_dict(),checkpoint_path)
  print("Training complete.")

if __name__ == "__main__":main()
