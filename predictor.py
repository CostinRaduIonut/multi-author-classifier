import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

class MultiWriterPredictor:

    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelul nu există: {model_path}")
        
        self.model = load_model(model_path)
        self.input_shape = self.model.input_shape[1:]  # Exclude batch dimension
        print(f"Model încărcat: {model_path}")
        print(f"Input shape așteptat: {self.input_shape}")
    
    def preprocess_image(self, image_path):
        """
        Preprocesează o imagine pentru predicție
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Nu se poate citi imaginea: {image_path}")
        
        h, w = img.shape
        target_h, target_w = self.input_shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                    cv2.BORDER_CONSTANT, value=255)
        
        normalized = padded.astype("float32") / 255.0
        processed = np.expand_dims(normalized, axis=-1)
        processed = np.expand_dims(processed, axis=0)
        
        return processed, padded
    
    def predict_single_image(self, image_path, show_image=True):
        processed_img, display_img = self.preprocess_image(image_path)
        prediction_proba = self.model.predict(processed_img, verbose=0)[0][0]
        prediction_class = int(prediction_proba > 0.5)
        
        if prediction_class == 0:
            result = "UN SINGUR AUTOR"
            confidence = (1 - prediction_proba) * 100
        else:
            result = "MAI MULȚI AUTORI"
            confidence = prediction_proba * 100
        
        print(f"\nREZULTAT PREDICȚIE:")
        print(f"Imagine: {os.path.basename(image_path)}")
        print(f"Predicție: {result}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Score raw: {prediction_proba:.4f}")
        
        if show_image:
            plt.figure(figsize=(10, 6))
            plt.imshow(display_img, cmap='gray')
            plt.title(f'Predicție: {result}\nConfidence: {confidence:.2f}%', 
                      fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return {
            'prediction_class': prediction_class,
            'prediction_proba': prediction_proba,
            'confidence': confidence,
            'result_text': result
        }

    def split_image_by_two_words(self, image_path):
        """
        Împarte o imagine în patch-uri care conțin câte 2 cuvinte.
        """
        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            raise ValueError(f"Imaginea nu poate fi încărcată: {image_path}")

        _, thresh = cv2.threshold(original_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        word_boxes = [cv2.boundingRect(c) for c in contours]
        word_boxes = sorted(word_boxes, key=lambda b: b[0])  # sortare stânga-dreapta

        patches = []
        for i in range(0, len(word_boxes), 2):
            group = word_boxes[i:i+2]
            if not group:
                continue
            x_min = min(b[0] for b in group)
            y_min = min(b[1] for b in group)
            x_max = max(b[0] + b[2] for b in group)
            y_max = max(b[1] + b[3] for b in group)

            crop = original_img[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue

            resized = cv2.resize(crop, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_AREA)
            normalized = resized.astype("float32") / 255.0
            processed = np.expand_dims(normalized, axis=-1)
            processed = np.expand_dims(processed, axis=0)
            patches.append((processed, crop))

        return patches

    def predict_by_two_word_patches(self, image_path, show_all=True):
        """
        Împarte imaginea în patch-uri (2 cuvinte) și rulează predicție pe fiecare.
        """
        print(f"\n🔍 Predicție pe patch-uri (2 cuvinte) pentru: {os.path.basename(image_path)}")
        patches = self.split_image_by_two_words(image_path)

        if not patches:
            print("⚠️ Nicio secțiune validă detectată.")
            return None

        results = []
        for i, (input_patch, raw_crop) in enumerate(patches):
            proba = self.model.predict(input_patch, verbose=0)[0][0]
            label = int(proba > 0.5)
            confidence = proba * 100 if label == 1 else (1 - proba) * 100
            result = "MAI MULȚI AUTORI" if label else "UN SINGUR AUTOR"

            results.append({
                "patch_index": i,
                "prediction_class": label,
                "prediction_proba": proba,
                "confidence": confidence,
                "result_text": result,
                "image": raw_crop
            })

        if show_all:
            cols = min(4, len(results))
            rows = (len(results) + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = axes.flatten() if len(results) > 1 else [axes]

            for i, r in enumerate(results):
                axes[i].imshow(r["image"], cmap='gray')
                axes[i].set_title(f"{r['result_text']}\nConf: {r['confidence']:.1f}%", fontsize=10)
                axes[i].axis("off")

            for j in range(i + 1, len(axes)):
                axes[j].axis("off")
            plt.tight_layout()
            plt.show()

        return results

    def predict_batch(self, image_paths):
        results = []
        print(f"\n🔄 Procesare {len(image_paths)} imagini...")
        for i, image_path in enumerate(image_paths, 1):
            try:
                print(f"\n[{i}/{len(image_paths)}] Procesare: {os.path.basename(image_path)}")
                result = self.predict_single_image(image_path, show_image=False)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"Eroare la {image_path}: {e}")
                continue

        single_writer_count = sum(1 for r in results if r['prediction_class'] == 0)
        multi_writer_count = len(results) - single_writer_count

        print(f"\nSUMAR REZULTATE:")
        print(f"Total imagini procesate: {len(results)}")
        print(f"Un singur autor: {single_writer_count}")
        print(f"Mai mulți autori: {multi_writer_count}")
        
        return results


def demo_prediction():
    
    # Găsește cel mai recent model
    model_files = [f for f in os.listdir('.') if f.startswith('multi_writer_detector_final_20250724_102957') and f.endswith('.h5')]
    
    if not model_files:
        print("Eroare la gasirea modelului")
        return
    
    # Ia cel mai recent model
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Ultimul model folosit: {latest_model}")
    
    # Inițializează predictorul
    try:
        predictor = MultiWriterPredictor(latest_model)
    except Exception as e:
        print(f"Eroare la incarcarea modelului: {e}")
        return
    
    # Exemplu de predicție (înlocuiește cu calea ta)
    test_image_path = "TRAIN_00003.jpg"  # Înlocuiește cu o imagine reală
    
    if os.path.exists(test_image_path):
        result = predictor.predict_single_image(test_image_path)
    else:
        print(f"Eroare la imaginea de test: {test_image_path}")

if __name__ == "__main__":
    demo_prediction()
