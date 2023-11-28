import numpy as np
import tensorflow as tf
from keras import layers, models, applications, optimizers, backend as K
# import tensorflow_hub as hub
import cv2
import math
# import imageio
from matplotlib import pyplot as plt
import random
import pandas as pd

frame_height = 180
frame_width = 180
num_channels = 3
seq_size = 32
lstm_units = 64

class FrameGenerator:
  def __init__(self, path, n_frames, training = False):
    self.path = path
    self.n_frames = n_frames
    self.training = training

  def __call__(self):
    video_path, label = self.path[0], self.path[1]
    pairs = list(zip(video_path, label))

    if self.training:
      random.shuffle(pairs)

    for path, label in pairs:
      # label = np.full((seq_size, 1), label)
      video_frames = frames_from_video_file(path, self.n_frames)
      yield video_frames, label

def format_frames(frame, output_size):

  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

def frames_from_video_file(video_path, n_frames, output_size = (frame_height,frame_width), frame_step = 1):

  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))
  max_frames_len = 500
  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  if n_frames == -1:
    need_length = video_length
  else:
    need_length = 1 + (n_frames - 1) * frame_step

  # introducing a limit to the number of frames to be read
  if need_length > max_frames_len:
    need_length = max_frames_len

  if int(need_length) > int(video_length):
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  result.append(format_frames(frame, output_size))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
        ret, frame = src.read()
        if not ret:
            break  # Exit the loop if there are no more frames
    if ret:
        frame = format_frames(frame, output_size)
        result.append(frame)
    else:
        break  # Exit the loop if there are no more frames

  # Repeat frames in the same order to fill the gap
  while len(result) < n_frames:
      for frame in result:
          result.append(frame)
          if len(result) == n_frames:
            break


  src.release()
  result = np.array(result)[..., [2, 1, 0]]

  return result


def load_and_preprocess(): 

  train_data = pd.read_csv("train.csv", header=None)
  test_data = pd.read_csv("test.csv", header=None)
  val_data = pd.read_csv("val.csv", header=None)

  print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

  # # Creating the training set
  output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                      tf.TensorSpec(shape = (), dtype = tf.float32))
  train_ds = tf.data.Dataset.from_generator(FrameGenerator(train_data, seq_size, training=True),
                                            output_signature = output_signature)
  # Creating the test set
  test_ds = tf.data.Dataset.from_generator(FrameGenerator(test_data, seq_size),
                                            output_signature = output_signature)
  # Creating the validation set
  val_ds = tf.data.Dataset.from_generator(FrameGenerator(val_data, seq_size),
                                            output_signature = output_signature)

  AUTOTUNE = tf.data.AUTOTUNE

  train_ds = train_ds.cache().prefetch(buffer_size = AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)
  test_ds = test_ds.cache().prefetch(buffer_size = AUTOTUNE)

  # for frames, labels in train_ds.take(5):
  #   print(frames.shape)
  #   print(labels)

  train_ds = train_ds.batch(10)
  val_ds = val_ds.batch(10)
  test_ds = test_ds.batch(10)

  return train_ds, val_ds, test_ds


# Step 4: Model Architecture
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

def build_model_effnet():
  effnet_model = applications.EfficientNetB0(include_top=False, weights='imagenet')
  effnet_model.trainable = False

  # Add the EfficientNet model
  model = models.Sequential()
  model.add(effnet_model)

  # Add a GlobalAveragePooling2D layer to reduce spatial dimensions
  model.add(layers.GlobalAveragePooling2D())

  # Reshape the output for LSTM input
  model.add(layers.Reshape((1, -1)))  # Flattening the feature maps

  # LSTM Part
  model.add(layers.LSTM(lstm_units, return_sequences=True))


  # model.add(layers.Conv2D(64, (3,3), activation='relu'))

  model.add(layers.Flatten())
  model.add(layers.Dense(128)) # 64

  # Output Layer

  model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification (deepfake or pristine)


  return model

def train_model(model, train_ds, val_ds, test_ds, model_name):
  
  # Compile the model
  model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])
  print("Compiling Complete")

  # Print the model summary
  model.summary()

  # Train the model
  print("Trainning started")
  hist = model.fit(train_ds, epochs=10, validation_data=val_ds)
  print("Training Complete")
  # Step 6: Model Evaluation
  loss, accuracy = model.evaluate(test_ds)
  print("Test Accuracy:", accuracy)
  print("Test Loss:", loss)

  # Save the model
  model.save("O:\Documents\MSc\Dissertation\Application\CNN-LSTM-Model\SavedModels\\"+model_name+"01_"+str(accuracy).replace(".","_")+".keras")

def plot_history(hist, model_name):
   
  # Plot training history
  plt.figure(figsize=(12, 6))

  # Plot accuracy
  plt.subplot(1, 2, 1)
  plt.plot(hist.history['accuracy'], label='Train')
  plt.plot(hist.history['val_accuracy'], label='Validation')
  plt.title(model_name+' - Model Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()

  # Plot loss
  plt.subplot(1, 2, 2)
  plt.plot(hist.history['loss'], label='Train')
  plt.plot(hist.history['val_loss'], label='Validation')
  plt.title(model_name+' - Model Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
'''
  # Plot F1 Score
  plt.subplot(1, 2, 3)
  plt.plot(hist.history['f1_m'], label='Train')
  plt.plot(hist.history['val_f1_m'], label='Validation')
  plt.title(model_name+' - Model F1 Score')
  plt.xlabel('Epoch')
  plt.ylabel('F1 Score')
  plt.legend()

  # Plot Precision
  plt.subplot(1, 2, 4)
  plt.plot(hist.history['precision_m'], label='Train')
  plt.plot(hist.history['val_precision_m'], label='Validation')
  plt.title(model_name+' - Model Precision')
  plt.xlabel('Epoch')
  plt.ylabel('Precision')
  plt.legend()

  # Plot Recall
  plt.subplot(1, 2, 5)
  plt.plot(hist.history['recall_m'], label='Train')
  plt.plot(hist.history['val_recall_m'], label='Validation')
  plt.title(model_name+' - Model Recall')
  plt.xlabel('Epoch')
  plt.ylabel('Recall')
  plt.legend()

  plt.tight_layout()
  plt.show()
'''

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def load_trained_model(mdl_path):
    # load trained model
    model = models.load_model(mdl_path)
    return model

def predict_video(video_path, n_frames=seq_size):
  # video_path = 'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\django\process\processed_vid\\'+vid_name
  xcept_mdl_path = 'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\CNN-LSTM-Model\SavedModels\\xception_06_ep50.keras'
  # resnet_model_path = 'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\CNN-LSTM-Model\SavedModels\model03_0_6253164410591125.keras'
  vgg16_model_path = 'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\CNN-LSTM-Model\SavedModels\\vgg16_model2_05_ep25.keras'
  meso_model_path = 'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\CNN-LSTM-Model\SavedModels\Meso_06_ep50.keras'
  model_xcept = load_trained_model(xcept_mdl_path)
  # model_resnet = load_trained_model(resnet_model_path)
  model_vgg16 = load_trained_model(vgg16_model_path)
  model_meso = load_trained_model(meso_model_path)

  frames = frames_from_video_file(video_path, n_frames) 
  frames = tf.expand_dims(frames, axis=0)

  vgg16_result = model_vgg16.predict(frames)
  vgg16_result = (math.ceil(vgg16_result[0]*1000))/1000

  meso_result = model_meso.predict(frames)
  meso_result = (math.ceil(meso_result[0]*1000))/1000

  xcept_result = model_xcept.predict(frames)
  xcept_result = (math.ceil(xcept_result[0]*1000))/1000

  # resnet_result = model_resnet.predict(frames)

  result = np.array([xcept_result, vgg16_result, meso_result])
  # result = np.mean(result, axis=1)
  result_round = np.round(result, decimals=3)
  print(result)
  return result_round

if __name__ == "__main__":
  train_ds, val_ds, test_ds = load_and_preprocess()
  print(train_ds.__len__)
  # xcept_model = build_model_xcept(train_ds, val_ds, test_ds)
  # train_model(xcept_model, train_ds, val_ds, test_ds, "Xception")
  # resnet_model = build_model_resnet()
  # train_model(resnet_model, train_ds, val_ds, test_ds, "ResNet")
  # mesonet_model = build_model_mesonet()
  # train_model(mesonet_model, train_ds, val_ds, test_ds, "MesoNet")
  # effnet_model = build_model_effnet()
  # train_model(effnet_model, train_ds, val_ds, test_ds, "EfficientNet")
  
