import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from load_process_data_torch import load_dataloader_multi_class
from models import ClassifierCNN_multi_class
import mlflow

def train_model_multi_class(N, win_width, slide, train_split, batch_size, num_epochs, learning_rate):

    train_loader, test_loader = load_dataloader_multi_class(win_width=win_width, slide=slide, N=N, batch_size=batch_size, train_split=train_split)

    model = ClassifierCNN_multi_class(N=N, win_width=win_width)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment("CNN Multi Class Experiment")
    with mlflow.start_run():

        mlflow.log_param("win_width", win_width)
        mlflow.log_param("slide", slide)
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
            # optimizer.zero_grad()
            for inputs, labels in train_loader:

                optimizer.zero_grad()
                outputs = model(inputs)
                targets = torch.argmax(labels, dim=1) 
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_total_loss += loss.item()
                samples += inputs.size(0)
            optimizer.step()
            ## *** eval ***
            model.eval()
            epoch_total_loss_eval = 0
            samples_eval = 0
            with torch.no_grad():
                all_preds = []
                all_labels = []
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    targets = torch.argmax(labels, dim=1) 
                    loss = criterion(outputs, targets)
                    epoch_total_loss_eval += loss.item()
                    samples_eval += inputs.size(0)

                    _, preds = torch.max(outputs, 1)
                    _, class_indices = torch.max(labels, 1)
                    all_preds.extend(preds.numpy())
                    all_labels.extend(class_indices.numpy())
                
            all_preds_np = np.concatenate([p.reshape(-1) for p in all_preds])
            all_labels_np = np.concatenate([l.reshape(-1) for l in all_labels])
            accuracy = np.sum(all_preds_np == all_labels_np) / len(all_labels_np)

            print(f"{epoch} avg loss: {epoch_total_loss/samples:0.5f}, avg eval_loss: {epoch_total_loss_eval/samples_eval:0.5f}, Eval Accuracy: {accuracy:.4f}")

            mlflow.log_metric("avg_loss_train", epoch_total_loss/samples, step=epoch)
            mlflow.log_metric("avg_loss_eval", epoch_total_loss_eval/samples_eval, step=epoch)
            mlflow.log_metric("accuracy_eval", accuracy, step=epoch)
                
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
        
        mlflow.pytorch.log_model(model, "model")
