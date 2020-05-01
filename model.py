from tensorflow.keras.models import Model
from tensorflow.keras.layers\
  import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import plot_model
from dataset import IMAGE_SIZE

def conv_net():
  inputs = Input(shape=(*IMAGE_SIZE, 1))
  
  hiddens = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
  hiddnes = MaxPool2D(pool_size=2)(hiddens)

  hiddens = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
  hiddnes = MaxPool2D(pool_size=2)(hiddnes)

  hiddens = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
  hiddnes = MaxPool2D(pool_size=2)(hiddnes)

  fcs = Flatten()(hiddens)
  fcs = Dense(units=512, activation='relu')(fcs)

  outputs = Dense(units=1, activation='sigmoid')(fcs)

  model = Model(inputs=inputs, outputs=outputs)

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

  # plot_model(model, to_file='model_plot.png')

  return model
