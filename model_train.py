# train_model.py

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set threading (optional tuning)
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(8)

# Directories
train_dir = 'deepfake_dataset/train'
val_dir = 'deepfake_dataset/val'

# Parameters
img_size = (224, 224)
batch_size = 16
AUTOTUNE = tf.data.AUTOTUNE

# Load and prepare training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode='binary',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)

# Load and prepare validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    label_mode='binary',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

# Data preprocessing and performance optimization
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)

train_ds = train_ds.cache('train_cache.tf-data').prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Load base model (Xception)
base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint('best_model_xception.h5', monitor='val_loss', save_best_only=True, verbose=1)

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stop, checkpoint]
)

# Save final model
model.save("final_deepfake_detector_model_xception.h5")
