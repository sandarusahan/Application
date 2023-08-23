import numpy as np
import tensorflow as tf
from keras import layers, models, applications, optimizers
import tensorflow_hub as hub
import cv2
import os
import glob
# import imageio
from matplotlib import pyplot as plt
import random
import pandas as pd

frame_height = 240
frame_width = 240
num_channels = 3
seq_size = 60
lstm_units = 64

class FrameGenerator:
  def __init__(self, path, n_frames, training = False):
    """ Returns a set of frames with their associated label. 

      Args:
        path: Video file paths.
        n_frames: Number of frames. 
        training: Boolean to determine if training dataset is being created.
    """
    self.path = path
    self.n_frames = n_frames
    self.training = training



  def __call__(self):
    video_path, label = self.path[0], self.path[1]

    pairs = list(zip(video_path, label))

    if self.training:
      random.shuffle(pairs)

    for path, label in pairs:
      video_frames = frames_from_video_file(path, self.n_frames) 
      yield video_frames, tf.expand_dims(label, axis=-1)

def frames_from_video_file(video_path, n_frames, output_size = (frame_height,frame_width), frame_step = 1):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))  

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  if n_frames == -1:
    need_length = video_length
  else:
    need_length = 1 + (n_frames - 1) * frame_step

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
    if ret:
      frame = format_frames(frame, output_size)
      result.append(frame)
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)[..., [2, 1, 0]]

  return result

def load_frames_from_video(video_path, index):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = format_frames(frame, (frame_height, frame_width))
        frames.append([index,frame])
    cap.release()
    return frames

def format_frames(frame, output_size):
  
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

def load_and_preprocess(): 

  # train_csv = open("D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\CNN-LSTM-Model\\train_sample.csv")
  # test_csv = open("D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\CNN-LSTM-Model\\test_sample.csv")
  # val_csv = open("D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\CNN-LSTM-Model\\val_sample.csv")

  # train_data = np.loadtxt(train_csv, delimeter=',', dtype=str)
  # test_data = np.loadtxt(test_csv, delimiter=',', dtype=str)
  # val_data = np.loadtxt(val_csv, delimiter=',', dtype=str)

  train_data = pd.read_csv("D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\\train2.csv", header=None)
  test_data = pd.read_csv("D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\\test2.csv", header=None)
  val_data = pd.read_csv("D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\\val2.csv", header=None)

  # train_data = pd.read_csv("O:\Documents\MSc\Dissertation\Application\\train1.csv", header=None)
  # test_data = pd.read_csv("O:\Documents\MSc\Dissertation\Application\\test1.csv", header=None)
  # val_data = pd.read_csv("O:\Documents\MSc\Dissertation\Application\\val1.csv", header=None)

  print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


  # # Create the training set
  output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                      tf.TensorSpec(shape = (None,), dtype = tf.float32))
  train_ds = tf.data.Dataset.from_generator(FrameGenerator(train_data, seq_size, training=True),
                                            output_signature = output_signature)
  # Create the test set
  test_ds = tf.data.Dataset.from_generator(FrameGenerator(test_data, seq_size),
                                            output_signature = output_signature)
  # Create the validation set
  val_ds = tf.data.Dataset.from_generator(FrameGenerator(val_data, seq_size),
                                            output_signature = output_signature)

  # AUTOTUNE = tf.data.AUTOTUNE

  # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
  # val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
  # print(train_ds)

  for frames, labels in train_ds.take(5):
    print(frames.shape)
    print(labels)

  return train_ds, val_ds, test_ds


# Step 4: Model Architecture
def train_model_xcept(train_ds, val_ds, test_ds):
  xcept_model = applications.Xception(include_top=False, weights='imagenet')
  xcept_model.trainable = False

  # Add the Xception model
  model = models.Sequential()
  model.add(xcept_model)

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

  # Compile the model
  model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])
  print("Compiling Complete")

  # Print the model summary
  model.summary()

  # Train the model
  print("Trainning started")
  hist = model.fit(train_ds, epochs=15, validation_data=val_ds)
  print("Training Complete")
  # Step 6: Model Evaluation
  loss, accuracy = model.evaluate(test_ds)
  print("Test Accuracy:", accuracy)
  print("Test Loss:", loss)

  # Save the model
  model.save("D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\CNN-LSTM-Model\SavedModels\\model04_"+str(accuracy).replace(".","_")+".keras")

  # Plot training history
  plt.figure(figsize=(12, 6))

  # Plot accuracy
  plt.subplot(1, 2, 1)
  plt.plot(hist.history['accuracy'], label='Train')
  plt.plot(hist.history['val_accuracy'], label='Validation')
  plt.title('Model Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()

  # Plot loss
  plt.subplot(1, 2, 2)
  plt.plot(hist.history['loss'], label='Train')
  plt.plot(hist.history['val_loss'], label='Validation')
  plt.title('Model Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()

  plt.tight_layout()
  plt.show()

  return model


def load_trained_model(mdl_path):
    # load trained model
    model = models.load_model(mdl_path)
    return model

def predict_video(video_path, n_frames=seq_size):
  # video_path = 'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\django\process\processed_vid\\'+vid_name
  mdl_path = 'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\CNN-LSTM-Model\SavedModels\model03_0_6253164410591125.keras'
  loaded_model = load_trained_model(mdl_path)
  frames = frames_from_video_file(video_path, n_frames) 
  result = loaded_model.predict(frames)
  print(result.mean())
  return result

if __name__ == "__main__":
  train_ds, val_ds, test_ds = load_and_preprocess()
  train_model_xcept(train_ds, val_ds, test_ds)