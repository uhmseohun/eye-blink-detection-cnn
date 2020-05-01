from model import conv_net
from dataset import load_data
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_val, y_val) = load_data()
model = conv_net()

train_datagen = ImageDataGenerator(
  rescale=1./255,
  rotation_range=15,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2
)

train_generator = train_datagen.flow(
  x=x_train, y=y_train,
  batch_size=32,
  shuffle=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow(
  x=x_val, y=y_val,
  batch_size=32,
  shuffle=True
)

checkpoint_path = './checkpoint/' + 'Epoch{epoch:04d}_Loss{val_loss:.4f}.h5'

save_callback = ModelCheckpoint(
  filepath=checkpoint_path,
  monitor='val_loss',
  verbose=1,
  save_best_only=False,
  save_weights_only=True,
  mode='auto',
  period=1
)

model.fit(
  train_generator,
  epochs=100,
  validation_data=val_generator,
  callbacks=[save_callback]
)

model.save_weights('trained_model/model_weights.h5')
