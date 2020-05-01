import numpy as np

IMAGE_SIZE = (26, 34)

def load_data ():
  x_train = np.load('dataset/train/x.npy').astype(np.float32)
  y_train = np.load('dataset/train/y.npy').astype(np.float32)

  x_val = np.load('dataset/val/x.npy').astype(np.float32)
  y_val = np.load('dataset/val/y.npy').astype(np.float32)

  return (x_train, y_train), (x_val, y_val)
