import os
import torch
from constants import *
from model import BigramModel
from tokenizers import Tokenizer
from train import prepare_datasets, train_model

# set seed for reproducibility
torch.manual_seed(SEED)

# create directory for storing checkpoints
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# pretraining phase
pretrain_file = os.path.join("pretrain_data", "processed.txt")
pretrain_train_set, pretrain_val_set = prepare_datasets(pretrain_file)

# initialize the model
model = BigramModel()

print("starting pretraining...")
model = train_model(
    model,
    pretrain_train_set,
    pretrain_val_set,
    EPOCHS_PRETRAIN,
    checkpoint_dir
)

# fine-tuning phase
finetune_file = os.path.join("train_data", "processed.txt")
finetune_train_set, finetune_val_set = prepare_datasets(finetune_file)

print("starting fine-tuning...")
model = train_model(
    model,
    finetune_train_set,
    finetune_val_set,
    EPOCHS_FINETUNE,
    checkpoint_dir,
    start_epoch=EPOCHS_PRETRAIN
)

# save the final fine-tuned model
final_model_path = "trained_model.pth"
torch.save(model.state_dict(), final_model_path)
print(f"final model saved at {final_model_path}")
