import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os

class MyDataset(Dataset):
    def __init__(self, data_arrays, labels, N):
        # 'data_arrays' is a list of your numpy arrays, e.g., [array1, array2, ...]
        # 'labels' is a list of corresponding labels, e.g., [label1, label2, ...]
        self.data = []
        self.labels = []

        for arr, label in zip(data_arrays, labels):

            num_samples = arr.shape[0] // N
            
            for i in range(num_samples):
                start_idx = i * N
                end_idx = start_idx + N
                sample = arr[start_idx:end_idx, :]
                self.data.append(torch.from_numpy(sample).float())
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def load_dataloader_binary(N, batch_size, train_split):

    MIN = -90
    MAX = 46

    folder_path = 'processed/bulk/'
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    labels = []
    arrays = []
    for file_path in file_paths:
        if "Time_Normal" in file_path:
            labels.append(0)
        else:
            labels.append(1)

        array = np.load(file_path)
        array = ((array - MIN) / (MAX - MIN) - 0.5) * 2
        arrays.append(array)

    full_dataset = MyDataset(arrays, labels, N)

    train_size = int(train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def load_dataloader_multi_class(win_width,slide, N, batch_size, train_split):

    MIN = -90
    MAX = 46

    folder_path = f'processed/{win_width}_{slide}/'
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # labels = []
    # arrays = []

    train_labels = []
    train_arrays = []
    test_labels = []
    test_arrays = []

    for file_path in file_paths:
        label = np.array([0, 0, 0, 0])
        if "Time_Normal" in file_path:
            label[0] = 1
        elif "B0" in file_path:
            label[1] = 1
        elif "IR" in file_path:
            label[2] = 1
        elif "OR" in file_path:
            label[3] = 1
        # labels.append(label)
        
        array = np.load(file_path)
        array = ((array - MIN) / (MAX - MIN) - 0.5) * 2

        split_idx = int(len(array) * train_split)

        test_arrays.append(array[:split_idx])
        test_labels.append(label)

        train_arrays.append(array[split_idx:])
        train_labels.append(label)

        # arrays.append(array)

    # full_dataset = MyDataset(arrays, labels, N)

    # train_size = int(train_split * len(full_dataset))
    # test_size = len(full_dataset) - train_size

    # train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_dataset = MyDataset(train_arrays, train_labels, N)
    test_dataset = MyDataset(test_arrays, test_labels, N)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# dataloader = load_dataloader(N=128, batch_size=32)

# for inputs, labels in dataloader:
#     print(inputs.shape, labels.shape)

## Find range to normalize by
# folder_path = 'processed/bulk/'
# file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
# MIN = -90
# MAX = 46
# arrays_min = 0
# arrays_max = 0
# for file_path in file_paths:
#     array = np.load(file_path)
#     array = ((array - MIN) / (MAX - MIN) - 0.5) * 2
#     if array.min() < arrays_min:
#         arrays_min = array.min()
#     if array.max() > arrays_max:
#         arrays_max = array.max()

# print(arrays_min, arrays_max)