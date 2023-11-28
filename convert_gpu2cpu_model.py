from keras import models, layers, applications, optimizers
import sys
sys.path.append('D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\CNN-LSTM-Model')
import model as m
vgg16_model_path = 'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\CNN-LSTM-Model\SavedModels\VGG16Weights\\vgg16_model2_05_ep25.h5'
xcept_model_path = 'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\CNN-LSTM-Model\SavedModels\XceptWeights\\xception_06_ep50.h5'
meso_model_path = 'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\CNN-LSTM-Model\SavedModels\MesoWeights\Meso_06_ep50.h5'

frame_height = 180
frame_width = 180
num_channels = 3
seq_size = 32
lstm_units = 64


def build_model_xcept():
  global model_name
  model_name = "xception"
  xcept_model = applications.Xception(include_top=False,
                                      weights='imagenet', input_shape = (frame_height, frame_width, 3))
  xcept_model.trainable = False

  # Add the ResNet model
  model = models.Sequential()
  # model.add(layers.Reshape((-1, fram, 3)))
  model.add(layers.Rescaling(scale=255))
  model.add(layers.TimeDistributed(xcept_model))

  # Add a GlobalAveragePooling2D layer to reduce spatial dimensions
  model.add(layers.GlobalAveragePooling3D())
  # Reshape the output for LSTM input
  model.add(layers.Reshape((seq_size, -1)))
  # LSTM Part
  model.add(layers.LSTM(lstm_units, return_sequences=True))

  model.add(layers.Dropout(0.3))

  model.add(layers.Dense(64)) # 64
  model.add(layers.Flatten())
  # Output Layer

  model.add(layers.Dense(1, activation='sigmoid'))
  # Binary classification (deepfake or pristine)

  input_shape = (None, seq_size, frame_width, frame_height, 3)
  model.build(input_shape)
  model.summary()

  return model

def build_model_resnet():
  global model_name
  model_name = "resnet50"
  resnet_model = applications.ResNet50(include_top=False,
                                      weights='imagenet', input_shape = (frame_height, frame_width, 3))
  resnet_model.trainable = False

  # Add the ResNet model
  model = models.Sequential()
  model.add(layers.TimeDistributed(resnet_model))

  # Add a GlobalAveragePooling2D layer
  # to reduce spatial dimensions
  model.add(layers.GlobalAveragePooling3D())
  # Reshape the output for LSTM input
  model.add(layers.Reshape((seq_size, -1)))
  # LSTM Part
  model.add(layers.LSTM(lstm_units, return_sequences=True))

  model.add(layers.Dropout(0.3))

  model.add(layers.Dense(128)) # 64
  model.add(layers.Flatten())
  # Output Layer

  model.add(layers.Dense(1, activation='sigmoid'))
  # Binary classification (deepfake or pristine)
  input_shape = (None, seq_size, frame_width, frame_height, 3)
  model.build(input_shape)
  model.summary()

  return model

def build_model_vgg16():
  global model_name
  model_name = "VGG16"
  vgg16_model = applications.VGG16(include_top=False,
                                      weights='imagenet', input_shape = (frame_height, frame_width, 3))
  vgg16_model.trainable = False

  # Add the ResNet model
  model = models.Sequential()
  model.add(layers.TimeDistributed(vgg16_model))

  # Add a GlobalAveragePooling2D layer
  # to reduce spatial dimensions
  model.add(layers.GlobalAveragePooling3D())

  # Reshape the output for LSTM input
  model.add(layers.Reshape((seq_size, -1)))  # Flattening the feature maps

  # LSTM Part
  model.add(layers.LSTM(lstm_units, return_sequences=True))


  model.add(layers.Dropout(0.3))

  model.add(layers.Flatten())
  model.add(layers.Dense(128)) # 64

  # Output Layer

  model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification (deepfake or pristine)
  input_shape = (None, seq_size, frame_width, frame_height, 3)
  model.build(input_shape)
  model.summary()

def build_model_mesonet(input_shape=(frame_height, frame_width, num_channels)):
  global model_name
  model_name = "Meso"
  model = models.Sequential()

  input_shape = (seq_size, frame_width, frame_height, num_channels)

  model.add(layers.Input(shape=input_shape))

  # Use TimeDistributed to apply Conv2D and MaxPooling2D to each frame
  model.add(layers.TimeDistributed(layers.Conv2D(8, (3, 3), padding='same', activation='relu')))
  model.add(layers.TimeDistributed(layers.BatchNormalization()))
  model.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), padding='same')))

  model.add(layers.TimeDistributed(layers.Conv2D(8, (5, 5), padding='same', activation='relu')))
  model.add(layers.TimeDistributed(layers.BatchNormalization()))
  model.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), padding='same')))

  model.add(layers.TimeDistributed(layers.Conv2D(16, (5, 5), padding='same', activation='relu')))
  model.add(layers.TimeDistributed(layers.BatchNormalization()))
  model.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), padding='same')))

  model.add(layers.TimeDistributed(layers.Conv2D(16, (5, 5), padding='same', activation='relu')))
  model.add(layers.TimeDistributed(layers.BatchNormalization()))
  model.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(4, 4), padding='same')))

  # Apply Flatten, Dropout, and Dense layers to each frame
  model.add(layers.TimeDistributed(layers.Flatten()))
  model.add(layers.TimeDistributed(layers.Dropout(0.5)))
  model.add(layers.TimeDistributed(layers.Dense(16)))
  model.add(layers.TimeDistributed(layers.LeakyReLU(alpha=0.1)))
  model.add(layers.TimeDistributed(layers.Dropout(0.5)))

  # Merge the temporal dimension (32 frames) using a GlobalAveragePooling1D
  model.add(layers.GlobalAveragePooling1D())

  # Final Dense layer for classification
  model.add(layers.Dense(1, activation='sigmoid'))

  model.summary()
  
  return model

# model = build_model_xcept()
# model = build_model_resnet()
model = build_model_mesonet()
# model = build_model_vgg16()
# model = models.load_model(vgg16_model_path)

model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(0.0001), metrics=['accuracy'])
# model.load_weights(vgg16_model_path)
# model.load_weights(xcept_model_path)
model.load_weights(meso_model_path)

# model.save("D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\CNN-LSTM-Model\SavedModels\\vgg16_model2_05_ep25.keras")
# model.save("D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\CNN-LSTM-Model\SavedModels\\xception_06_ep50.keras")
model.save("D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\CNN-LSTM-Model\SavedModels\\Meso_06_ep50.keras")
# train_ds, val_ds, test_ds = m.load_and_preprocess()

# loss, acc = model.evaluate(test_ds)

# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
# print("Restored model, loss: {:5.2f}%".format(100*loss))