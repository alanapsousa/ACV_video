"""
Script d'entraînement SIMPLE - Version minimaliste
Entraîne un modèle rapidement avec ton dataset
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_images():
    """Charger les images du dataset"""
    print(" Chargement des images...")
    
    UP_DIR = "./dataset/up"
    DOWN_DIR = "./dataset/down"
    IMG_SIZE = 224
    
    images = []
    labels = []
    
    # Charger images UP (label=1)
    for filename in os.listdir(UP_DIR):
        if filename.endswith(('.jpg', '.png')):
            filepath = os.path.join(UP_DIR, filename)
            img = cv2.imread(filepath)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype('float32') / 255.0
                images.append(img)
                labels.append(1)
    
    # Charger images DOWN (label=0)
    for filename in os.listdir(DOWN_DIR):
        if filename.endswith(('.jpg', '.png')):
            filepath = os.path.join(DOWN_DIR, filename)
            img = cv2.imread(filepath)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype('float32') / 255.0
                images.append(img)
                labels.append(0)
    
    X = np.array(images)
    y = np.array(labels)
    
    print(f"✓ {len(X)} images chargées")
    
    return X, y

def main():
    print("=" * 60)
    print(" ENTRAÎNEMENT SIMPLE DU MODÈLE")
    print("=" * 60 + "\n")
    
    # Charger les données
    X, y = load_images()
    
    if len(X) < 20:
        print(" Pas assez d'images! (minimum 20)")
        return
    
    print(f"✓ UP: {np.sum(y)} images")
    print(f"✓ DOWN: {np.sum(y == 0)} images\n")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f" Train: {len(X_train)} images")
    print(f" Test: {len(X_test)} images\n")
    
    # Créer le modèle
    print(" Création du modèle...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    # Architecture simple
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Modèle créé\n")
    
    # Entraîner
    print(" Entraînement (10 epochs)...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=8,
        verbose=1
    )
    
    # Évaluation
    print("\n Évaluation...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"✓ Accuracy test: {accuracy:.2%}")
    print(f"✓ Loss test: {loss:.4f}\n")
    
    # Sauvegarder
    os.makedirs("./models", exist_ok=True)
    model.save("./models/pushup_model.h5")
    
    print("=" * 60)
    print(" MODÈLE ENTRAÎNÉ ET SAUVEGARDÉ!")
    print("=" * 60)
    print("\nProchaine étape:")
    print("  python 03_pushup_counter.py")

if __name__ == "__main__":
    main()
