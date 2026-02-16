import numpy as np
import matplotlib.pyplot as plt
from model_handwriting_features import build_multi_writer_model, compile_model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    TensorBoard, CSVLogger)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import os
from datetime import datetime
import tensorflow as tf

def create_callbacks(model_name="multi_writer_detector"):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),

        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1,
            cooldown=3
        ),

        ModelCheckpoint(
            f'best_{model_name}_{timestamp}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='max'
        ),

        TensorBoard(
            log_dir=f'logs/{model_name}_{timestamp}',
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        ),

        CSVLogger(f'training_log_{model_name}_{timestamp}.csv', append=True)
    ]

    return callbacks, timestamp

def create_data_augmentation():

    datagen = ImageDataGenerator(
        rotation_range=2,           # Rotație foarte ușoară
        width_shift_range=0.06,     # Shift orizontal mic
        height_shift_range=0.04,    # Shift vertical mic
        shear_range=0.04,          # Shear foarte ușor
        zoom_range=0.03,           # Zoom foarte ușor
        horizontal_flip=False,      # NU flip pentru text
        vertical_flip=False,        # NU flip pentru text
        fill_mode='constant',        # Umple pixelii noi cu o valoare constantă
        cval=1.0                   # Valoarea de umplere (alb, deoarece imaginile sunt normalizate la 0-1)
    )
    return datagen

def evaluate_model_detailed(model, X_test, y_test, model_name="model"):

    print(f"\n{'='*50}")
    print(f"EVALUARE DETALIATĂ - {model_name.upper()}")
    print(f"{'='*50}")

    # Predicții
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Metrici de bază
    accuracy = np.mean(y_pred == y_test)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    print(f"REZULTATE GENERALE:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"Total samples: {len(y_test)}")

    # Distribuția claselor
    unique, counts = np.unique(y_test, return_counts=True)
    print(f"\nDISTRIBUȚIA CLASELOR în test set:")
    for cls, count in zip(unique, counts):
        label = "Single-writer" if cls == 0 else "Multi-writer"
        print(f"   • {label}: {count} samples ({count/len(y_test)*100:.1f}%)")

    # Classification report
    print(f"\nCLASSIFICATION REPORT:")
    target_names = ['Single-writer', 'Multi-writer']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Confusion matrix
    print(f"CONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Predicted")
    print(f"Single Multi")
    print(f"Actual Single [{cm[0,0]:6d} {cm[0,1]:6d}]")
    print(f"Multi [{cm[1,0]:6d} {cm[1,1]:6d}]")

    # Analiza erorilor
    false_positives = np.sum((y_pred == 1) & (y_test == 0))
    false_negatives = np.sum((y_pred == 0) & (y_test == 1))

    print(f"\nANALIZA ERORILOR:")
    print(f"False Positives: {false_positives} (single-writer clasificat ca multi-writer)")
    print(f"False Negatives: {false_negatives} (multi-writer clasificat ca single-writer)")

    return {
        'accuracy': accuracy,
        'auc': auc_score,
        'predictions': y_pred_proba,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    }

def main():
    """
    Funcția principală pentru training și evaluare
    """

    # === 1. ÎNCARCĂ DATELE ===
    print("\n[1/6] Încărcare date...")

    # Încearcă să încarci datele din noul fișier generat de preprocessing, care nu are scurgere de date
    data_files = [
        "./preprocessed_data.npz"
    ]

    data_loaded = False
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"Găsit: {data_file}")
            data = np.load(data_file)
            data_loaded = True
            break
    if not data_loaded:
        print("Nu s-a găsit niciun fișier de date!")
        print("Rulează mai întâi scriptul de preprocessing (improved_dataset_preparation.py)!")
        return

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test'] if 'X_test' in data else None
    y_test = data['y_test'] if 'y_test' in data else None


    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape} - Single: {np.sum(y_train==0)}, Multi: {np.sum(y_train==1)}")
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val.shape} - Single: {np.sum(y_val==0)}, Multi: {np.sum(y_val==1)}")
    if X_test is not None:
        print(f"X_test: {X_test.shape}")
        print(f"y_test: {y_test.shape} - Single: {np.sum(y_test==0)}, Multi: {np.sum(y_test==1)}")


    # === 2. CONSTRUIEȘTE MODELUL ===
    print("\n[2/6]Construire model...")

    input_shape = X_train.shape[1:]  # Ia forma din datele reale
    print(f"Input shape: {input_shape}")

    model = build_multi_writer_model(input_shape=input_shape)
    model = compile_model(model, learning_rate=0.001)

    print(f"Model creat: {model.name}")
    print(f"Total parametri: {model.count_params():,}")

    # === 3. PREGĂTEȘTE TRAINING-UL ===
    print("\n[3/6] Pregătire training...")

    callbacks, timestamp = create_callbacks("multi_writer_detector")
    datagen = create_data_augmentation()

    # Fit data augmentation pe datele de training
    datagen.fit(X_train)

    print(f"Callbacks create")
    print(f"Data augmentation pregătită")
    print(f"Timestamp: {timestamp}")

    # === 4. ANTRENEAZĂ MODELUL ===
    print("\n[4/6]Antrenare model...")

    # Calculează steps
    batch_size = 64
    epochs = 5
    # steps_per_epoch = len(X_train) // batch_size
    # steps_per_epoch = 100
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    # print(f"Steps per epoch: {steps_per_epoch}")

    # Training cu data augmentation
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        # steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    print(f"Training completat!")

    # === 5. EVALUEAZĂ MODELUL ===
    print("\n[5/6] Evaluare model...")

    # Evaluare pe validation set
    val_results = evaluate_model_detailed(model, X_val, y_val, "VALIDATION")

    # Evaluare pe test set (dacă există)
    if X_test is not None:
        test_results = evaluate_model_detailed(model, X_test, y_test, "TEST")
    else:
        test_results = None
        print("Setul de testare nu a fost încărcat. Nu se poate efectua evaluarea pe test.")


    # === 6. SALVEAZĂ MODELUL ===
    print("\n[6/6] Salvare model...")

    final_model_name = f"multi_writer_detector_final_{timestamp}.h5"
    model.save(final_model_name)

    print(f"Model salvat: {final_model_name}")

    # === SUMAR FINAL ===
    print(f"\n{'='*60}")
    print("TRAINING COMPLETAT CU SUCCES!")
    print(f"{'='*60}")
    print(f"Model salvat: {final_model_name}")
    print(f"Accuracy finală (Validation): {val_results['accuracy']:.4f} ({val_results['accuracy']*100:.2f}%)")
    print(f"AUC Score (Validation): {val_results['auc']:.4f}")
    if test_results:
        print(f"Accuracy finală (Test): {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
        print(f"AUC Score (Test): {test_results['auc']:.4f}")
    print(f"Logs disponibile în: logs/multi_writer_detector_{timestamp}/")
    print(f"Training log: training_log_multi_writer_detector_{timestamp}.csv")

    # Plotează rezultatele
    plot_training_history(history, timestamp)

    return model, history, val_results, test_results

def plot_training_history(history, timestamp):

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History - Multi-Writer Detection', fontsize=16)

    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    
    if 'auc' in history.history:
        axes[1, 0].plot(history.history['auc'], label='Training AUC')
        axes[1, 0].plot(history.history['val_auc'], label='Validation AUC')
        axes[1, 0].set_title('Model AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    if 'precision' in history.history:
        axes[1, 1].plot(history.history['precision'], label='Training Precision')
        axes[1, 1].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 1].set_title('Model Precision')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(f'training_history_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Graficele salvate în: training_history_{timestamp}.png")

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    model, history, val_results, test_results = main()
