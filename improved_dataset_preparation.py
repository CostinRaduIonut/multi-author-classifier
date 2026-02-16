# # E OK - initial
# import numpy as np
# import pandas as pd
# import os
# import cv2
# import random
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
# from tqdm import tqdm


# class SimpleImagePreprocessor:
#     def __init__(self, target_size=(256, 128)):
#         self.target_size = target_size
    
#     def resize_with_padding(self, img):
#         """
#         Simplu și sigur - fără complicații
#         """
#         h, w = img.shape[:2]
#         target_w, target_h = self.target_size
        
#         # Scale pentru a păstra aspect ratio
#         scale = min(target_w / w, target_h / h)
#         new_w, new_h = int(w * scale), int(h * scale)
        
#         # Resize simplu
#         resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
#         # Padding
#         pad_w = target_w - new_w
#         pad_h = target_h - new_h
#         top = pad_h // 2
#         bottom = pad_h - top
#         left = pad_w // 2
#         right = pad_w - left
        
#         padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
#                                    cv2.BORDER_CONSTANT, value=255)
#         return padded
    
#     def enhance_simple(self, img):
#         """
#         Îmbunătățire simplă - fără PIL
#         """
#         # Contrast simplu
#         enhanced = cv2.convertScaleAbs(img, alpha=1.1, beta=0)
        
#         # Sharpening ușor
#         kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
#         sharpened = cv2.filter2D(enhanced, -1, kernel)
        
#         # Blend
#         result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
#         return result
    
#     def process_image(self, img_path):
#         """
#         Procesare simplă și sigură
#         """
#         try:
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#             if img is None:
#                 return None
            
#             # Îmbunătățire
#             img = self.enhance_simple(img)
            
#             # Resize
#             img = self.resize_with_padding(img)
            
#             # Normalize
#             img = img.astype("float32") / 255.0
            
#             return img
#         except:
#             return None

# # === CONFIGURAȚIA TA - SCHIMBĂ AICI PATHS ===
# IMG_TRAIN_PATH = "./dataset/test_v2/test"
# IMG_VALID_PATH = "./dataset/validation_v2/validation"
# IMG_TEST_PATH = "./dataset/test_v2/test"
# CSV_TRAIN_PATH = "./dataset/written_name_test_v2.csv"
# CSV_VALID_PATH = "./dataset/written_name_validation_v2.csv"
# CSV_TEST_PATH = "./dataset/written_name_test_v2.csv"

# IMG_SIZE = (256, 128)
# MIN_SAMPLES_PER_ID = 4
# MULTI_WRITER_RATIO = 0.1  # Redus pentru siguranță

# preprocessor = SimpleImagePreprocessor(target_size=IMG_SIZE)

# # === 1. Load data ===
# def load_csv_with_split(path, split_name):
#     df = pd.read_csv(path)
#     df["split"] = split_name
#     return df

# print("[1/7] Loading CSV files...")
# df_train = load_csv_with_split(CSV_TRAIN_PATH, "train")
# df_val = load_csv_with_split(CSV_VALID_PATH, "val")
# df_test = load_csv_with_split(CSV_TEST_PATH, "test")
# df = pd.concat([df_train, df_val, df_test], ignore_index=True)
# df.dropna(subset=["FILENAME", "IDENTITY"], inplace=True)
# print(f"   Loaded {len(df)} records")

# # === 2. Resolve paths ===
# def resolve_path(filename):
#     for path in [IMG_TRAIN_PATH, IMG_VALID_PATH, IMG_TEST_PATH]:
#         full_path = os.path.join(path, filename)
#         if os.path.exists(full_path):
#             return full_path
#     return None

# print("[2/7] Resolving image paths...")
# df["image_path"] = df["FILENAME"].apply(resolve_path)
# df = df[df["image_path"].notnull()]
# print(f"   Found {len(df)} valid images")

# # === 3. Filter authors ===
# print("[3/7] Filtering authors...")
# df = df.groupby("IDENTITY").filter(lambda x: len(x) >= MIN_SAMPLES_PER_ID)
# print(f"   Kept {len(df)} images from authors with {MIN_SAMPLES_PER_ID}+ samples")

# # === 4. Label encoding ===
# print("[4/7] Encoding labels...")
# le = LabelEncoder()
# df["label_id"] = le.fit_transform(df["IDENTITY"])
# num_authors = len(le.classes_)
# print(f"   {num_authors} unique authors")

# # === 5. Group by author ===
# print("[5/7] Grouping by author...")
# author_to_images = {}
# for _, row in df.iterrows():
#     author = row['label_id']
#     path = row['image_path']
#     if author not in author_to_images:
#         author_to_images[author] = []
#     author_to_images[author].append(path)

# # === 6. Process splits ===
# def process_split_safe(df_split, split_name):
#     """
#     VERSIUNE SIGURĂ - cu limite și verificări
#     """
#     print(f"\n[6/7] Processing {split_name} split...")
    
#     # Filtrează autorii din acest split
#     authors_in_split = df_split['label_id'].unique()
#     split_author_images = {}
    
#     for author in authors_in_split:
#         if author in author_to_images:
#             split_author_images[author] = author_to_images[author]
    
#     print(f"   Authors in {split_name}: {len(split_author_images)}")
    
#     X = []
#     y = []
    
#     # === SINGLE-WRITER SAMPLES ===
#     print(f"   Processing single-writer images...")
#     total_single = sum(len(paths) for paths in split_author_images.values())
    
#     processed = 0
#     with tqdm(total=total_single, desc="Single-writer") as pbar:
#         for author, paths in split_author_images.items():
#             for img_path in paths:
#                 img = preprocessor.process_image(img_path)
#                 if img is not None:
#                     X.append(np.expand_dims(img, axis=-1))
#                     y.append(0)  # Single-writer label
                
#                 processed += 1
#                 pbar.update(1)
                
#                 # SIGURANȚĂ: Limitează la 10000 imagini per split
#                 if processed >= 10000:
#                     print(f"   LIMIT: Stopped at {processed} images for safety")
#                     break
#             if processed >= 10000:
#                 break
    
#     print(f"   Processed {len(X)} single-writer images")
    
#     # === MULTI-WRITER SAMPLES ===
#     num_multi_samples = min(int(len(X) * MULTI_WRITER_RATIO))
#     authors_list = list(split_author_images.keys())
    
#     if len(authors_list) >= 2: 
#         print(f"   Creating {num_multi_samples} multi-writer samples...")
        
#         multi_created = 0
#         attempts = 0
#         max_attempts = num_multi_samples * 3  # SIGURANȚĂ: max încercări
        
#         with tqdm(total=num_multi_samples, desc="Multi-writer") as pbar:
#             while multi_created < num_multi_samples and attempts < max_attempts:
#                 attempts += 1
                
#                 try:
#                     # Alege 2 autori random
#                     if len(authors_list) >= 2:
#                         chosen_authors = random.sample(authors_list, 2)
                        
#                         # Alege câte o imagine de la fiecare
#                         img_paths = []
#                         for author in chosen_authors:
#                             if author in split_author_images and split_author_images[author]:
#                                 img_path = random.choice(split_author_images[author])
#                                 img_paths.append(img_path)
                        
#                         # Verifică că avem 2 imagini
#                         if len(img_paths) == 2:
#                             # Procesează imaginile
#                             imgs = []
#                             for path in img_paths:
#                                 img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#                                 if img is not None:
#                                     img = preprocessor.enhance_simple(img)
#                                     imgs.append(img)
                            
#                             # Creează composite dacă avem 2 imagini valide
#                             if len(imgs) == 2:
#                                 # Resize la aceeași înălțime
#                                 target_h = IMG_SIZE[1]
#                                 resized_imgs = []
#                                 for img in imgs:
#                                     h, w = img.shape
#                                     scale = target_h / h
#                                     new_w = int(w * scale)
#                                     resized = cv2.resize(img, (new_w, target_h))
#                                     resized_imgs.append(resized)
                                
#                                 # Concatenează
#                                 composite = np.hstack(resized_imgs)
                                
#                                 # Resize final
#                                 composite = preprocessor.resize_with_padding(composite)
#                                 composite = composite.astype("float32") / 255.0
                                
#                                 X.append(np.expand_dims(composite, axis=-1))
#                                 y.append(1)  # Multi-writer label
                                
#                                 multi_created += 1
#                                 pbar.update(1)
                
#                 except Exception as e:
#                     continue  # Ignoră erorile și continuă
        
#         print(f"   Created {multi_created} multi-writer samples")
#     else:
#         print(f"   Skipped multi-writer (not enough authors)")
    
#     return np.array(X), np.array(y)

# # Procesează train și validation
# train_df = df[df['split'] == 'train']
# val_df = df[df['split'] == 'val']

# X_train, y_train = process_split_safe(train_df, "TRAIN")
# X_val, y_val = process_split_safe(val_df, "VALIDATION")

# # === 7. Save results ===
# print(f"\n[7/7] Saving results...")
# SAVE_PATH = "./preprocessed_data_FIXED.npz"
# np.savez_compressed(SAVE_PATH, 
#                    X_train=X_train, y_train=y_train, 
#                    X_val=X_val, y_val=y_val)

# print(f"\n" + "="*50)
# print("✅ PROCESSING COMPLETE - NO INFINITE LOOPS!")
# print("="*50)
# print(f"✅ Saved to: {SAVE_PATH}")
# print(f"✅ Train set: {X_train.shape}")
# print(f"   - Single-writer: {np.sum(y_train == 0)}")
# print(f"   - Multi-writer: {np.sum(y_train == 1)}")
# print(f"✅ Val set: {X_val.shape}")
# print(f"   - Single-writer: {np.sum(y_val == 0)}")
# print(f"   - Multi-writer: {np.sum(y_val == 1)}")

# # Vizualizare rapidă
# print(f"\n📊 Showing sample results...")
# fig, axes = plt.subplots(2, 4, figsize=(16, 8))
# fig.suptitle('Fixed Processing Results - No Infinite Loops!', fontsize=16)

# for i in range(30):
#     row = i // 4
#     col = i % 4
    
#     if i < len(X_train):
#         img = X_train[i].squeeze()
#         label_text = "Single-writer" if y_train[i] == 0 else "Multi-writer"
#         axes[row, col].imshow(img, cmap='gray')
#         axes[row, col].set_title(f"{label_text}")
#         axes[row, col].axis("off")
#     else:
#         axes[row, col].axis("off")

# plt.tight_layout()
# plt.show()

# import numpy as np
# import pandas as pd
# import os
# import cv2
# import random
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# random.seed(42)
# np.random.seed(42)

# class SimpleImagePreprocessor:
#     def __init__(self, target_size=(256, 128)):
#         self.target_size = target_size

#     def resize_with_padding(self, img):
#         h, w = img.shape[:2]
#         target_w, target_h = self.target_size
#         scale = min(target_w / w, target_h / h)
#         new_w, new_h = int(w * scale), int(h * scale)
#         resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
#         pad_w = target_w - new_w
#         pad_h = target_h - new_h
#         top = pad_h // 2
#         bottom = pad_h - top
#         left = pad_w // 2
#         right = pad_w - left
#         padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
#                                     cv2.BORDER_CONSTANT, value=255) 
#         return padded

#     def enhance_simple(self, img):
#         enhanced = cv2.convertScaleAbs(img, alpha=1.1, beta=0)
#         kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
#         sharpened = cv2.filter2D(enhanced, -1, kernel)
#         result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
#         return result

#     def process_image(self, img_path):
#         try:
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#             if img is None:
#                 return None
#             img = self.enhance_simple(img)
#             img = self.resize_with_padding(img)
#             img = img.astype("float32") / 255.0
#             return img
#         except Exception as e:
#             with open("errors.log", "a") as f:
#                 f.write(f"[process_image] {img_path}: {e}\n")
#             return None


# IMG_TRAIN_PATH = "./dataset/train_v2/train" 
# IMG_VALID_PATH = "./dataset/validation_v2/validation"
# IMG_TEST_PATH = "./dataset/test_v2/test" 

# CSV_TRAIN_PATH = "./dataset/written_name_train_v2.csv" 
# CSV_VALID_PATH = "./dataset/written_name_validation_v2.csv"
# CSV_TEST_PATH = "./dataset/written_name_test_v2.csv" 

# IMG_SIZE = (256, 128) 
# MIN_SAMPLES_PER_ID = 4
# MULTI_WRITER_RATIO = 0.8 

# preprocessor = SimpleImagePreprocessor(target_size=IMG_SIZE)

# def load_csv_with_split(path, split_name):
#     df = pd.read_csv(path)
#     df["split"] = split_name
#     return df

# print("[1/7] Loading CSV files...")
# df_train = load_csv_with_split(CSV_TRAIN_PATH, "train")
# df_val = load_csv_with_split(CSV_VALID_PATH, "val")
# df_test = load_csv_with_split(CSV_TEST_PATH, "test") 

# df = pd.concat([df_train, df_val, df_test], ignore_index=True)
# df.dropna(subset=["FILENAME", "IDENTITY"], inplace=True)
# print(f"   Loaded {len(df)} records")

# def resolve_path(filename):
#     for path in [IMG_TRAIN_PATH, IMG_VALID_PATH, IMG_TEST_PATH]:
#         full_path = os.path.join(path, filename)
#         if os.path.exists(full_path):
#             return full_path
#     return None

# print("[2/7] Resolving image paths...")
# df["image_path"] = df["FILENAME"].apply(resolve_path)
# df = df[df["image_path"].notnull()]
# print(f"   Found {len(df)} valid images")

# print("[3/7] Filtering authors...")
# df = df.groupby("IDENTITY").filter(lambda x: len(x) >= MIN_SAMPLES_PER_ID)
# print(f"   Kept {len(df)} images from authors with {MIN_SAMPLES_PER_ID}+ samples")

# print("[4/7] Encoding labels...")
# le = LabelEncoder()
# df["label_id"] = le.fit_transform(df["IDENTITY"])
# num_authors = len(le.classes_)
# print(f"   {num_authors} unique authors")

# print("\n[!] Verificare suprapunere autori între seturi...")
# train_authors = set(df[df['split'] == 'train']['label_id'].unique())
# val_authors = set(df[df['split'] == 'val']['label_id'].unique())
# test_authors = set(df[df['split'] == 'test']['label_id'].unique())

# overlap_train_val = train_authors.intersection(val_authors)
# overlap_train_test = train_authors.intersection(test_authors)
# overlap_val_test = val_authors.intersection(test_authors)

# if overlap_train_val:
#     print(f"   AVERTISMENT: Autori comuni între seturile TRAIN și VALIDATION! ({len(overlap_train_val)} autori)")
# if overlap_train_test:
#     print(f"   AVERTISMENT CRITIC: Autori comuni între seturile TRAIN și TEST! ({len(overlap_train_test)} autori)")
#     print("   Aceasta este o cauză majoră a supraînvățării și a metricilor de performanță înșelătoare.")
# if overlap_val_test:
#     print(f"   AVERTISMENT: Autori comuni între seturile VALIDATION și TEST! ({len(overlap_val_test)} autori)")
# if not (overlap_train_val or overlap_train_test or overlap_val_test):
#     print("   Nu s-au detectat suprapuneri de autori între seturi. Excelent!")

# print("[5/7] Grouping by author...")
# author_to_images = {}
# for _, row in df.iterrows():
#     author = row['label_id']
#     path = row['image_path']
#     if author not in author_to_images:
#         author_to_images[author] = []
#     author_to_images[author].append(path)

# def process_split_safe(df_split, split_name):
#     print(f"\n[6/7] Processing {split_name} split...")
#     authors_in_split = df_split['label_id'].unique()
#     split_author_images = {a: author_to_images[a] for a in authors_in_split if a in author_to_images}
#     print(f"   Authors in {split_name}: {len(split_author_images)}")

#     X, y = [], []

#     print(f"   Processing single-writer images...")
#     total_single = sum(len(paths) for paths in split_author_images.values())
#     processed = 0
#     with tqdm(total=total_single, desc=f"Single-writer ({split_name})") as pbar:
#         for author, paths in split_author_images.items():
#             for img_path in paths:
#                 img = preprocessor.process_image(img_path)
#                 if img is not None:
#                     X.append(np.expand_dims(img, axis=-1))
#                     y.append(0) 
#                 processed += 1
#                 pbar.update(1)
#                 if processed >= 10000 and split_name == "TRAIN": 
#                     print(f"   LIMIT: Stopped at {processed} images for safety in {split_name}")
#                     break
#             if processed >= 10000 and split_name == "TRAIN":
#                 break

#     print(f"   Processed {len(X)} single-writer images")

#     num_multi_samples = int(len(X) * MULTI_WRITER_RATIO)
#     authors_list = list(split_author_images.keys())

#     if len(authors_list) >= 2:
#         print(f"   Creating {num_multi_samples} multi-writer samples...")
#         multi_created = 0
#         attempts = 0
#         max_attempts = num_multi_samples * 2
#         with tqdm(total=num_multi_samples, desc=f"Multi-writer ({split_name})") as pbar:
#             while multi_created < num_multi_samples and attempts < max_attempts:
#                 attempts += 1
#                 try:
#                     chosen_authors = random.sample(authors_list, 2)
#                     img_paths = []
#                     for author in chosen_authors:
#                         if author in split_author_images and split_author_images[author]:
#                             img_path = random.choice(split_author_images[author])
#                             img_paths.append(img_path)

#                     if len(img_paths) == 2:
#                         imgs = []
#                         for path in img_paths:
#                             img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#                             if img is not None:
#                                 img = preprocessor.enhance_simple(img)
#                                 imgs.append(img)

#                         if len(imgs) == 2:
                           
#                             target_h = IMG_SIZE[1]
#                             resized_imgs = []
#                             for img in imgs:
#                                 h, w = img.shape
#                                 scale = target_h / h
#                                 new_w = int(w * scale)
#                                 resized = cv2.resize(img, (new_w, target_h))
#                                 resized_imgs.append(resized)

#                             composite = np.hstack(resized_imgs)
                           
#                             composite = preprocessor.resize_with_padding(composite).astype("float32") / 255.0
#                             X.append(np.expand_dims(composite, axis=-1))
#                             y.append(1) # Label 1 for multi-writer
#                             multi_created += 1
#                             pbar.update(1)
#                 except Exception as e:
#                     with open("errors.log", "a") as f:
#                         f.write(f"[multi-writer] {e}\n")
#                     continue
#         if multi_created < num_multi_samples:
#             print(f"   Warning: Could not create all {num_multi_samples} multi-writer samples. Created {multi_created}.")
#         print(f"   Created {multi_created} multi-writer samples")
#     else:
#         print(f"   Skipped multi-writer (not enough authors in {split_name})")

#     return np.array(X), np.array(y)

# # Împărțim dataframe-ul filtrat înapoi în seturile originale
# train_df = df[df['split'] == 'train']
# val_df = df[df['split'] == 'val']
# # df_test nu este folosit direct aici pentru a genera X_test, y_test, dar este important pentru verificarea suprapunerii.
# # Dacă vrei să generezi și X_test, y_test, ar trebui să apelezi process_split_safe și pentru df_test.

# X_train, y_train = process_split_safe(train_df, "TRAIN")
# X_val, y_val = process_split_safe(val_df, "VALIDATION")

# print(f"\n[7/7] Saving results...")
# SAVE_PATH = "./preprocessed_data.npz"
# np.savez_compressed(SAVE_PATH,
#                     X_train=X_train, y_train=y_train,
#                     X_val=X_val, y_val=y_val)

# print(f"\n" + "="*50)
# print("PROCESSING COMPLETE - NO INFINITE LOOPS!")
# print("="*50)
# print(f"Saved to: {SAVE_PATH}")
# print(f"Train set: {X_train.shape}")
# print(f"   - Single-writer: {np.sum(y_train == 0)}")
# print(f"   - Multi-writer: {np.sum(y_train == 1)}")
# print(f"Val set: {X_val.shape}")
# print(f"   - Single-writer: {np.sum(y_val == 0)}")
# print(f"   - Multi-writer: {np.sum(y_val == 1)}")

# print(f"\nShowing sample results...")
# fig, axes = plt.subplots(2, 4, figsize=(16, 8))
# fig.suptitle('Fixed Processing Results - No Infinite Loops!', fontsize=16)
# for i in range(8):
#     if i < len(X_train):
#         img = X_train[i].squeeze()
#         label_text = "Single-writer" if y_train[i] == 0 else "Multi-writer"
#         axes[i//4, i%4].imshow(img, cmap='gray')
#         axes[i//4, i%4].set_title(label_text)
#         axes[i//4, i%4].axis("off")
#     else:
#         axes[i//4, i%4].axis("off")
# plt.tight_layout()
# plt.show()

import numpy as np
import pandas as pd
import os
import cv2
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

class SimpleImagePreprocessor:
    def __init__(self, target_size=(256, 128)):
        self.target_size = target_size

    def resize_with_padding(self, img):
        h, w = img.shape[:2]
        target_w, target_h = self.target_size
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
        return padded

    def enhance_simple(self, img):
        enhanced = cv2.convertScaleAbs(img, alpha=1.1, beta=0)
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        return result

    def process_image(self, img_path):
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            img = self.enhance_simple(img)
            img = self.resize_with_padding(img)
            img = img.astype("float32") / 255.0
            return img
        except Exception as e:
            with open("errors.log", "a") as f:
                f.write(f"[process_image] {img_path}: {e}\n")
            return None

IMG_TRAIN_PATH = "./dataset/train_v2/train" 
IMG_VALID_PATH = "./dataset/validation_v2/validation"
IMG_TEST_PATH = "./dataset/test_v2/test" 

CSV_TRAIN_PATH = "./dataset/written_name_train_v2.csv" 
CSV_VALID_PATH = "./dataset/written_name_validation_v2.csv"
CSV_TEST_PATH = "./dataset/written_name_test_v2.csv" 

IMG_SIZE = (256, 128)
MIN_SAMPLES_PER_ID = 4
MULTI_WRITER_RATIO = 1.0

preprocessor = SimpleImagePreprocessor(target_size=IMG_SIZE)

def load_csv_with_split(path, split_name):
    df = pd.read_csv(path)
    df["split"] = split_name
    return df

print("[1/7] Loading CSV files...")
df_train = load_csv_with_split(CSV_TRAIN_PATH, "train")
df_val = load_csv_with_split(CSV_VALID_PATH, "val")
df_test = load_csv_with_split(CSV_TEST_PATH, "test")
df = pd.concat([df_train, df_val, df_test], ignore_index=True)
df.dropna(subset=["FILENAME", "IDENTITY"], inplace=True)
print(f"   Loaded {len(df)} records")

def resolve_path(filename):
    for path in [IMG_TRAIN_PATH, IMG_VALID_PATH, IMG_TEST_PATH]:
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            return full_path
    return None

print("[2/7] Resolving image paths...")
df["image_path"] = df["FILENAME"].apply(resolve_path)
df = df[df["image_path"].notnull()]
print(f"   Found {len(df)} valid images")

print("[3/7] Filtering authors...")
df = df.groupby("IDENTITY").filter(lambda x: len(x) >= MIN_SAMPLES_PER_ID)
print(f"   Kept {len(df)} images from authors with {MIN_SAMPLES_PER_ID}+ samples")

print("[4/7] Encoding labels...")
le = LabelEncoder()
df["label_id"] = le.fit_transform(df["IDENTITY"])
num_authors = len(le.classes_)
print(f"   {num_authors} unique authors")

print("\n[4.1] Re-split intern în 80-10-10 din TRAIN original...")
original_train_df = df[df["split"] == "train"]
author_ids = original_train_df["IDENTITY"].unique()
train_authors, temp_authors = train_test_split(author_ids, test_size=0.2, random_state=42)
val_authors, test_authors = train_test_split(temp_authors, test_size=0.5, random_state=42)

def split_internal(author):
    if author in train_authors:
        return "train_internal"
    elif author in val_authors:
        return "val_internal"
    else:
        return "test_internal"

df.loc[df["split"] == "train", "split"] = df[df["split"] == "train"]["IDENTITY"].apply(split_internal)
print("   - train_internal:", len(df[df["split"] == "train_internal"]))
print("   - val_internal:", len(df[df["split"] == "val_internal"]))
print("   - test_internal:", len(df[df["split"] == "test_internal"]))

print("\n[!] Verificare suprapunere autori între seturi...")
train_authors_set = set(df[df['split'] == 'train_internal']['label_id'].unique())
val_authors_set = set(df[df['split'] == 'val_internal']['label_id'].unique())
test_authors_set = set(df[df['split'] == 'test_internal']['label_id'].unique())

if train_authors_set & val_authors_set:
    print("   AVERTISMENT: autori comuni între TRAIN și VAL")
if train_authors_set & test_authors_set:
    print("   AVERTISMENT CRITIC: autori comuni între TRAIN și TEST")
if val_authors_set & test_authors_set:
    print("   AVERTISMENT: autori comuni între VAL și TEST")
if not (train_authors_set & val_authors_set or train_authors_set & test_authors_set or val_authors_set & test_authors_set):
    print("   Nu s-au detectat suprapuneri de autori între splituri interne. Excelent!")

print("[5/7] Grouping by author...")
author_to_images = {}
for _, row in df.iterrows():
    author = row['label_id']
    path = row['image_path']
    if author not in author_to_images:
        author_to_images[author] = []
    author_to_images[author].append(path)

def process_split_safe(df_split, split_name):
    print(f"\n[6/7] Processing {split_name} split...")
    authors_in_split = df_split['label_id'].unique()
    split_author_images = {a: author_to_images[a] for a in authors_in_split if a in author_to_images}
    print(f"   Authors in {split_name}: {len(split_author_images)}")

    X, y = [], []

    print(f"   Processing single-writer images...")
    total_single = sum(len(paths) for paths in split_author_images.values())
    processed = 0
    with tqdm(total=total_single, desc=f"Single-writer ({split_name})") as pbar:
        for author, paths in split_author_images.items():
            for img_path in paths:
                img = preprocessor.process_image(img_path)
                if img is not None:
                    X.append(np.expand_dims(img, axis=-1))
                    y.append(0)
                processed += 1
                pbar.update(1)
                if processed >= 10000 and split_name == "train_internal":
                    print(f"   LIMIT: Stopped at {processed} images for safety in {split_name}")
                    break
            if processed >= 10000 and split_name == "train_internal":
                break

    print(f"   Processed {len(X)} single-writer images")

    num_multi_samples = int(len(X) * MULTI_WRITER_RATIO)
    authors_list = list(split_author_images.keys())

    if len(authors_list) >= 2:
        print(f"   Creating {num_multi_samples} multi-writer samples...")
        multi_created = 0
        attempts = 0
        max_attempts = num_multi_samples * 2
        with tqdm(total=num_multi_samples, desc=f"Multi-writer ({split_name})") as pbar:
            while multi_created < num_multi_samples and attempts < max_attempts:
                attempts += 1
                try:
                    chosen_authors = random.sample(authors_list, 2)
                    img_paths = []
                    for author in chosen_authors:
                        if author in split_author_images and split_author_images[author]:
                            img_path = random.choice(split_author_images[author])
                            img_paths.append(img_path)

                    if len(img_paths) == 2:
                        imgs = []
                        for path in img_paths:
                            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                img = preprocessor.enhance_simple(img)
                                imgs.append(img)

                        if len(imgs) == 2:
                            target_h = IMG_SIZE[1]
                            resized_imgs = [cv2.resize(img, (int(img.shape[1] * target_h / img.shape[0]), target_h)) for img in imgs]
                            composite = np.hstack(resized_imgs)
                            composite = preprocessor.resize_with_padding(composite).astype("float32") / 255.0
                            X.append(np.expand_dims(composite, axis=-1))
                            y.append(1)
                            multi_created += 1
                            pbar.update(1)
                except Exception as e:
                    with open("errors.log", "a") as f:
                        f.write(f"[multi-writer] {e}\n")
        print(f"   Created {multi_created} multi-writer samples")
    else:
        print(f"   Skipped multi-writer (not enough authors in {split_name})")

    return np.array(X), np.array(y)

train_df = df[df['split'] == 'train_internal']
val_df = df[df['split'] == 'val_internal']

X_train, y_train = process_split_safe(train_df, "train_internal")
X_val, y_val = process_split_safe(val_df, "val_internal")

print(f"\n[7/7] Saving results...")
SAVE_PATH = "./preprocessed_data.npz"
np.savez_compressed(SAVE_PATH, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

print(f"\n{'='*50}")
print("PROCESSING COMPLETE - NO INFINITE LOOPS!")
print("="*50)
print(f"Saved to: {SAVE_PATH}")
print(f"Train set: {X_train.shape} | Single: {np.sum(y_train == 0)} | Multi: {np.sum(y_train == 1)}")
print(f"Val set: {X_val.shape} | Single: {np.sum(y_val == 0)} | Multi: {np.sum(y_val == 1)}")

print("\nShowing sample results...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Sample Results - train_internal', fontsize=16)
for i in range(8):
    if i < len(X_train):
        img = X_train[i].squeeze()
        label_text = "Single-writer" if y_train[i] == 0 else "Multi-writer"
        axes[i//4, i%4].imshow(img, cmap='gray')
        axes[i//4, i%4].set_title(label_text)
        axes[i//4, i%4].axis("off")
    else:
        axes[i//4, i%4].axis("off")
plt.tight_layout()
plt.show()
