import tensorflow as tf
import pandas as pd

# === 1. Set GPU 6 only ===
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 6:
    try:
        tf.config.set_visible_devices(gpus[6], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[6], True)
        print("Using GPU 6")
    except RuntimeError as e:
        print(e)
else:
    print("GPU 6 not available, using default device.")

# === 2. Your DataFrame with audio file paths and labels ===
file_paths = df['filepath'].tolist()
labels = df['label'].tolist()  # Optional if you want to evaluate accuracy

# === 3. Preprocessing function to load audio ===
def load_audio(path, label):
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    desired_length = 16000
    audio = audio[:desired_length]
    paddings = [[0, tf.maximum(0, desired_length - tf.shape(audio)[0])]]
    audio = tf.pad(audio, paddings)
    return audio, label

# === 4. Create batched dataset ===
dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
dataset = dataset.map(load_audio, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# === 5. Load your model (assuming function build_model exists) ===
num_classes = len(set(labels))
model = build_model(input_shape=(16000,), num_classes=num_classes)

# If you have saved weights, load them here
# model.load_weights('path_to_weights.h5')

# === 6. Run inference ===
predictions = model.predict(dataset)

# If you want predicted class indices:
predicted_classes = tf.argmax(predictions, axis=1).numpy()

print(predicted_classes)
