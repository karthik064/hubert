import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Assume these are defined:
# y_true (N,), integer labels 0..22
# y_scores (N, 23), predicted probabilities per class
# class_names (list of 23 class names)

# 1. Binarize labels for multi-class ROC
y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))

# 2. Create directory to save plots
save_dir = "outputs/roc_curves"
os.makedirs(save_dir, exist_ok=True)

# 3. Plot setup
plt.figure(figsize=(14, 10))

for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
    roc_auc = auc(fpr, tpr)
    
    # Smooth interpolation (optional)
    fpr_smooth = np.linspace(0, 1, 1000)
    tpr_smooth = np.interp(fpr_smooth, fpr, tpr)
    
    plt.plot(fpr_smooth, tpr_smooth, lw=2,
             label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

# 4. Plot formatting
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curves')
plt.legend(loc='lower right', fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout()

# 5. Save the figure
save_path = os.path.join(save_dir, "multi_class_roc.png")
plt.savefig(save_path, dpi=300)
plt.close()

print(f"ROC curves saved to: {save_path}")import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# Assume these are already defined
# y_true, y_pred, y_scores
# class_names = ['class_0', 'class_1', ..., 'class_22']

# 1. Create a directory to save the plot
save_dir = "outputs/confusion_matrix"
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# 2. Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# 3. Normalize for smoother visual (optional)
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

# 4. Plot
plt.figure(figsize=(16, 12))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='YlGnBu',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='gray', cbar=True)

plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=90)  # X-axis vertical
plt.yticks(rotation=0)
plt.tight_layout()

# 5. Save the plot
save_path = os.path.join(save_dir, "confusion_matrix.png")
plt.savefig(save_path, dpi=300)
plt.close()  # Close the figure to free memory

print(f"Confusion matrix saved to: {save_path}")
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
