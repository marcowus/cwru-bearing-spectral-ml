from train_multi_class_utils import train_model_multi_class

train_split = 0.80
batch_size = 32
num_epochs = 100
learning_rate = 0.001

N = 32
win_width=524
slide=128

print(f"**** {N} {win_width} {slide} ****")
train_model_multi_class(N, win_width, slide, train_split, batch_size, num_epochs, learning_rate)
