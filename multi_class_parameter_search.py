from train_multi_class import train_model_multi_class

train_split = 0.80
batch_size = 32
num_epochs = 100
learning_rate = 0.001

## win_width search: win_width = number of timesteps to use in bin when computing fft. Also equal to frequency resolution each time step.
# N = 256
# win_width=1024
# slide=256
# print(f"**** {N} {win_width} {slide} ****")
# train_model_multi_class(N, win_width, slide, train_split, batch_size, num_epochs, learning_rate)

# N = 128
# win_width=1024
# slide=256
# print(f"**** {N} {win_width} {slide} ****")
# train_model_multi_class(N, win_width, slide, train_split, batch_size, num_epochs, learning_rate)

# N = 64
# win_width=1024
# slide=256
# print(f"**** {N} {win_width} {slide} ****")
# train_model_multi_class(N, win_width, slide, train_split, batch_size, num_epochs, learning_rate)

# N = 32
# win_width=1024
# slide=256
# print(f"**** {N} {win_width} {slide} ****")
# train_model_multi_class(N, win_width, slide, train_split, batch_size, num_epochs, learning_rate)

# N = 16
# win_width=1024
# slide=256
# print(f"**** {N} {win_width} {slide} ****")
# train_model_multi_class(N, win_width, slide, train_split, batch_size, num_epochs, learning_rate)

# N search: N = number of bins to use as input
N = 32
win_width=2048
slide=512
print(f"**** {N} {win_width} {slide} ****")
train_model_multi_class(N, win_width, slide, train_split, batch_size, num_epochs, learning_rate)

N = 32
win_width=1024
slide=256
print(f"**** {N} {win_width} {slide} ****")
train_model_multi_class(N, win_width, slide, train_split, batch_size, num_epochs, learning_rate)

N = 32
win_width=524
slide=128
print(f"**** {N} {win_width} {slide} ****")
train_model_multi_class(N, win_width, slide, train_split, batch_size, num_epochs, learning_rate)

N = 32
win_width=128
slide=16
print(f"**** {N} {win_width} {slide} ****")
train_model_multi_class(N, win_width, slide, train_split, batch_size, num_epochs, learning_rate)