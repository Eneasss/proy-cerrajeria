import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models

#Variables Globales
DATASET_PATH = os.path.join("dataset")
MODEL_PATH = os.path.join("model")
os.makedirs(MODEL_PATH, exist_ok=True)

IMG_SIZE = (320, 320)
BACTH_SIZE = 16
EPOCHS = 10
FINE_TUNE_EPOCHS = 8
TOTAL_EPOCHS = EPOCHS + FINE_TUNE_EPOCHS
LEARNING_RATE = 0.001
FINE_TUNE_LR = 0.00001

#Division de datos en entrenamiento y validacion en 80% | 20%
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BACTH_SIZE,
    crop_to_aspect_ratio=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BACTH_SIZE,
    crop_to_aspect_ratio=True
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\n Clases detectadas ({num_classes}): {class_names}")

with open(os.path.join(MODEL_PATH, "labels.json"), "w") as f:
    json.dump(class_names, f)

# Optimización de pipeline de I/O en memoria
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 3. DATA AUGMENTATION (En memoria)
# ==========================================
# Simula variaciones de posicionamiento y luz en el local
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),  # Giro de hasta +/- ~28 grados
    layers.RandomZoom(0.05),
    layers.RandomContrast(0.1)
], name="data_augmentation")

# ==========================================
# 4. ARQUITECTURA: TRANSFER LEARNING
# ==========================================
# Carga de MobileNetV2 preentrenado sin la capa final de ImageNet
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Congelar capas convolucionales base

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
# Normalización específica requerida por MobileNetV2 (escala [-1, 1])
x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

# ==========================================
# 5. Fase 1 Entrenamiento de cabecera
# ==========================================

print("\n" + "=" * 50)
print(">>> FASE 1: Entrenando cabecera con base congelada...")
print("=" * 50)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

#model.summary()

# ==========================================
# 5. FASE 2: FINE-TUNING (Ajuste fino)
# ==========================================

print("\n" + "=" * 50)
print(">>> FASE 2: Descongelando últimas capas para Fine-Tuning...")
print("=" * 50)

base_model.trainable = True

fine_tune_at = len(base_model.layers) - 60
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

trainable_count = len([l for l in base_model.layers if l.trainable])
print(f"Capas base totales: {len(base_model.layers)} | Capas descongeladas: {trainable_count}")

# 3. Recompilar obligatoriamente con Learning Rate 100 veces menor
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)
callbacks_finetune = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]

# Continuar entrenamiento desde la última época completada
initial_epoch_phase2 = len(history_phase1.epoch)

history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=TOTAL_EPOCHS,
    initial_epoch=initial_epoch_phase2,
    callbacks=callbacks_finetune
)

""" # ==========================================
# 5. ENTRENAMIENTO
# ==========================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )
] """

# Guardar modelo final en formato Keras nativo
keras_path = os.path.join(MODEL_PATH, "key_classifier.keras")
model.save(keras_path)
print(f"\nModelo final guardado en: {keras_path}")

# ==========================================
# 6. CONVERSIÓN Y CUANTIZACIÓN A TFLITE
# ==========================================

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = os.path.join(MODEL_PATH, "key_classifier.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
print(f"Modelo TFLite exportado: {tflite_path} ({size_mb:.2f} MB)")

""" print("\nIniciando entrenamiento...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Guardar modelo Keras original (.keras)
keras_model_path = os.path.join(MODEL_PATH, "key_classifier.keras")
model.save(keras_model_path)
print(f"\nModelo base guardado en: {keras_model_path}") """

""" # ==========================================
# 6. CONVERSIÓN Y CUANTIZACIÓN A TFLITE
# ==========================================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Optimización de rango dinámico para CPU
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = os.path.join(MODEL_PATH, "key_classifier.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
print(f"Modelo TFLite generado exitosamente: {tflite_path} ({size_mb:.2f} MB)") """