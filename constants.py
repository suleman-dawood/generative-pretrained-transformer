# constants for training and model configuration
DATA_SPLIT = 0.85  # train-val split
NUM_MERGES = 100000  # vocab size
NUM_LINE_SPLIT = 10000  # number of lines to split
MAX_CONTEXT_LENGTH = 1000  # maximum context length

# seed and training parameters
SEED = 7  # seed for reproducibility
LR = 0.0001  # learning rate, controls how quickly the model converges
EPOCHS_PRETRAIN = 5000  # number of iterations for pretraining
EPOCHS_FINETUNE = 2000  # number of iterations for fine-tuning
DROPOUT = 0.2  # regularization parameter to prevent overfitting
WEIGHT_DECAY = 0.001  # weight decay for regularization
GRAD_CLIP = 1.0  # gradient clipping to prevent exploding gradients
SAVE_STEPS = 200  # save model every x number of steps

# model configuration
context_size = 128  # size of the input context window
batch_size = 32  # number of samples in each batch
val_iterations = 100  # how often to compare train and validation loss
num_embeddings = 384  # size of embedding vectors
head_count = 6  # number of attention heads in multi-head attention
layer_count = 6  # number of layers in the model
