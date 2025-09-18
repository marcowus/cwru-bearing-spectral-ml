import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from load_process_data_torch import load_dataloader_binary
from models import ClassifierCNN_binary

import mlflow

N = 128 # Number of freq samples per input

train_split = 0.80
batch_size = 32
num_epochs = 10
learning_rate = 0.001

train_loader, test_loader = load_dataloader_binary(N=N, batch_size=batch_size, train_split=train_split)

model = ClassifierCNN_binary()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

mlflow.set_experiment("CNN Binary Experiment")
with mlflow.start_run():

    mlflow.log_param("N", N)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("train_split", train_split)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("learning_rate", learning_rate)

    for epoch in range(num_epochs):
        ## *** Train ***
        model.train()
        epoch_total_loss = 0
        samples = 0
        for inputs, labels in train_loader:

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))

            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()
            samples += inputs.size(0)
        
        ## *** eval ***
        model.eval()
        epoch_total_loss_eval = 0
        samples_eval = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels.float().unsqueeze(1))
                epoch_total_loss_eval += loss.item()
                samples_eval += inputs.size(0)

        print(f"{epoch} avg loss: {epoch_total_loss/samples}, avg eval_loss: {epoch_total_loss_eval/samples_eval}")

        mlflow.log_metric("avg_loss_train", epoch_total_loss/samples, step=epoch)
        mlflow.log_metric("avg_loss_eval", epoch_total_loss_eval/samples_eval, step=epoch)
    
    mlflow.pytorch.log_model(model, "model")