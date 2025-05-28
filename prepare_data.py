import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def count_images(folder):
    return len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

img_size = (224, 224)
num_real = count_images('/home/Desktop/Deepfake/Dataset/Real')
num_fake = count_images('/home/Desktop/Deepfake/Dataset/Fake')
total = num_real + num_fake

# Pre-allocate arrays using memory-mapped files
images = np.memmap('images.dat', dtype='float32', mode='w+', shape=(total, 224, 224, 3))
labels = np.memmap('labels.dat', dtype='int32', mode='w+', shape=(total,))

def load_images(folder, label, start_index):
    idx = start_index
    for filename in os.listdir(folder):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            img = img / 255.0
            images[idx] = img
            labels[idx] = label
            idx += 1
    return idx

next_index = load_images('/home/Desktop/Deepfake/Dataset/Real', 0, 0)
next_index = load_images('/home/Desktop/Deepfake/Dataset/Fake', 1, next_index)

# Convert to normal numpy arrays (will be backed by disk if needed)
images = np.array(images)
labels = np.array(labels)

# Shuffle and split
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

np.savez("dataset_split.npz", train_images=train_images, test_images=test_images,
         train_labels=train_labels, test_labels=test_labels)

data = np.load("dataset_split.npz")
train_images = data['train_images']
train_labels = data['train_labels']
test_images = data['test_images']
test_labels = data['test_labels']

# Output directory
base_dir = 'deepfake_dataset'
os.makedirs(base_dir, exist_ok=True)

# Helper function to save images
def save_images(images, labels, split):
    for i, (img, label) in enumerate(tqdm(zip(images, labels), total=len(images), desc=f"Saving {split}")):
        label_dir = os.path.join(base_dir, split, 'Real' if label == 0 else 'Fake')
        os.makedirs(label_dir, exist_ok=True)
        img_path = os.path.join(label_dir, f'{split}_{i}.jpg')
        img_bgr = (img * 255).astype(np.uint8)
        cv2.imwrite(img_path, cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR))

# Save training and test images
save_images(train_images, train_labels, 'train')
save_images(test_images, test_labels, 'val')
