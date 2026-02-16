from improved_dataset_preparation import df, le, preprocess_image
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm

MODEL_PATH = "./multi_writer_detector_final_20250724_102957.h5"  # modifică dacă e cazul

# === 1. Selectează doar datele de test ===
test_df = df[df["split"] == "test"]
X_test = np.array([preprocess_image(p) for p in tqdm(test_df["image_path"], desc="Preproc Test")])
y_test = np.array(test_df["label_id"])

# === 2. Încarcă modelul și evaluează ===
model = load_model(MODEL_PATH)
loss, acc = model.evaluate(X_test, y_test, verbose=0)

# === 3. Clasificare + raport ===
y_probs = model.predict(X_test)
y_pred = np.argmax(y_probs, axis=1)

report = classification_report(y_test, y_pred, target_names=le.classes_)

# === 4. Scrie în fișier ===
with open("test_results.txt", "w", encoding="utf-8") as f:
    f.write(f"Test Loss: {loss:.4f}\n")
    f.write(f"Test Accuracy: {acc:.4f}\n\n")
    f.write("=== Classification Report ===\n")
    f.write(report)

print("[INFO] Testare completă. Rezultatele au fost scrise în test_results.txt.")
