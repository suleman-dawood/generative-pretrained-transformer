# GPT-from-scratch
Creating a basic GPT model from scratch

This code implements a transformer-based language model inspired by modern architectures like GPT. Its goal is to train a neural network to predict and generate text based on the works of Friedrich Nietzsche. By learning the relationships between characters and their contexts, the model can produce new, contextually relevant sequences of text. For now it only has the encoder so it somply tries to generate text

# Bigram Language Model

This repository contains code for training and fine-tuning a Bigram-based language model using PyTorch. The model is capable of generating coherent text based on learned token sequences. The process includes data preprocessing, tokenization, training, and model checkpoints.

## Features

- **Pretraining and Fine-tuning**: The model is first pretrained on a large dataset and then fine-tuned on a specific corpus.
- **Text Generation**: Once trained, the model can generate coherent text based on a given context.
- **Model Checkpoints**: Save and resume training with checkpoints, ensuring robustness.
- **Data Preprocessing**: The pipeline handles data preparation, tokenization, and ensures the proper format for training.

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/bigram-language-model.git
   cd bigram-language-model

    Install dependencies:

    pip install -r requirements.txt

    Prepare your dataset for pretraining and fine-tuning.

Training
Pretraining

Run the following command to start pretraining:

python pretrain.py

Fine-tuning

After pretraining, fine-tune the model using:

python finetune.py

Text Generation

To generate text based on a trained model, run:

python generate_text.py

This will generate text and save it to output.txt.
Files

    pretrain.py: Script to handle pretraining.
    finetune.py: Script for fine-tuning the model.
    generate_text.py: Script to generate text from the trained model.
    train.py: Contains training logic.
    model.py: Defines the architecture of the Bigram model.

License

This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgements

    The model and architecture were inspired by bigram models and attention mechanisms in neural networks.
    Thanks to the open-source community for their contributions to PyTorch and related libraries.