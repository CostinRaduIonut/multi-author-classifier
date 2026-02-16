# import numpy as np
# import pandas as pd
# import os
# import cv2
# import random
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import threading

# # === CONFIG ===
# IMG_TRAIN_PATH = "./dataset/test_v2/test"
# IMG_VALID_PATH = "./dataset/validation_v2/validation"
# IMG_TEST_PATH  = "./dataset/test_v2/test"

# CSV_TRAIN_PATH = "./dataset/written_name_test_v2.csv"
# CSV_VALID_PATH = "./dataset/written_name_validation_v2.csv"
# CSV_TEST_PATH  = "./dataset/written_name_test_v2.csv"

# IMG_SIZE = (128, 64)
# MIN_SAMPLES_PER_ID = 4
# MULTI_WRITER_AUTHORS = 2
# MULTI_WRITER_RATIO = 0.5

# # === 1. Load CSVs and merge ===
# def load_csv_with_split(path, split_name):
#     df = pd.read_csv(path)
#     df["split"] = split_name
#     return df

# df_train = load_csv_with_split(CSV_TRAIN_PATH, "train")
# df_val   = load_csv_with_split(CSV_VALID_PATH, "val")
# df_test  = load_csv_with_split(CSV_TEST_PATH, "test")

# df = pd.concat([df_train, df_val, df_test], ignore_index=True)
# df.dropna(subset=["FILENAME", "IDENTITY"], inplace=True)

# # === 2. Resolve full image paths ===
# def resolve_path(filename):
#     for path in [IMG_TRAIN_PATH, IMG_VALID_PATH, IMG_TEST_PATH]:
#         full_path = os.path.join(path, filename)
#         if os.path.exists(full_path):
#             return full_path
#     return None

# df["image_path"] = df["FILENAME"].apply(resolve_path)
# df = df[df["image_path"].notnull()]

# # === 3. Filter authors with enough samples ===
# df = df.groupby("IDENTITY").filter(lambda x: len(x) >= MIN_SAMPLES_PER_ID)

# # === 4. Label encoding ===
# le = LabelEncoder()
# df["label_id"] = le.fit_transform(df["IDENTITY"])

# # === 5. Resize with padding ===
# def resize_with_padding(img, target_size=IMG_SIZE):
#     h, w = img.shape[:2]
#     scale = min(target_size[1] / h, target_size[0] / w)
#     new_w, new_h = int(w * scale), int(h * scale)
#     resized = cv2.resize(img, (new_w, new_h))
#     pad_w = target_size[0] - new_w
#     pad_h = target_size[1] - new_h
#     top = pad_h // 2
#     bottom = pad_h - top
#     left = pad_w // 2
#     right = pad_w - left
#     padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
#     return padded

# # === 6. Preprocess image ===
# def preprocess_image(img_path):
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise FileNotFoundError(f"Could not read image: {img_path}")
#     img = resize_with_padding(img, IMG_SIZE)
#     img = img.astype("float32") / 255.0
#     return img

# # === 7. Group images by author ===
# author_to_images = {}
# for _, row in df.iterrows():
#     author = row['label_id']
#     path = row['image_path']
#     author_to_images.setdefault(author, []).append(path)

# # === 8. Prepare train and val DataFrames ===
# train_df = df[df['split'] == 'train']
# val_df = df[df['split'] == 'val']

# def prepare_split(df_split):
#     authors_in_split = df_split['label_id'].unique()
#     split_author_images = {a: author_to_images[a] for a in authors_in_split if a in author_to_images}

#     X = []
#     y = []

#     # Single-writer samples (label=0)
#     for author, paths in split_author_images.items():
#         for p in paths:
#             img = preprocess_image(p)
#             img = np.expand_dims(img, axis=-1)
#             X.append(img)
#             y.append(0)

#     # Multi-writer samples (label=1)
#     num_multi_samples = int(len(X) * MULTI_WRITER_RATIO)
#     authors_list = list(split_author_images.keys())
#     random.shuffle(authors_list)

#     for _ in range(num_multi_samples):
#         chosen_authors = random.sample(authors_list, MULTI_WRITER_AUTHORS)
#         imgs = []
#         for a in chosen_authors:
#             img_path = random.choice(split_author_images[a])
#             img = preprocess_image(img_path)
#             imgs.append(img)
#         composite = np.hstack(imgs)
#         composite_resized = resize_with_padding((composite * 255).astype(np.uint8), IMG_SIZE)
#         composite_resized = composite_resized.astype("float32") / 255.0
#         composite_resized = np.expand_dims(composite_resized, axis=-1)
#         X.append(composite_resized)
#         y.append(1)

#     X = np.array(X)
#     y = np.array(y)
#     return X, y

# X_train, y_train, X_val, y_val = None, None, None, None

# def load_train():
#     global X_train, y_train
#     print("[THREAD] Preparing training data...")
#     X_train, y_train = prepare_split(train_df)

# def load_val():
#     global X_val, y_val
#     print("[THREAD] Preparing validation data...")
#     X_val, y_val = prepare_split(val_df)

# # Threads
# train_thread = threading.Thread(target=load_train)
# val_thread = threading.Thread(target=load_val)

# train_thread.start()
# val_thread.start()

# train_thread.join()
# val_thread.join()

# # === Save to .npz ===
# SAVE_PATH = "./preprocessed_data.npz"
# np.savez_compressed(SAVE_PATH, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
# print(f"[INFO] Saved preprocessed data to {SAVE_PATH}")

# # === 10. Print dataset info ===
# print(f"[INFO] Train set: {X_train.shape}, Labels distribution: {np.bincount(y_train)}")
# print(f"[INFO] Val set: {X_val.shape}, Labels distribution: {np.bincount(y_val)}")

# # === Optional: Visualize some images ===
# for i in range(7):
#     img = X_train[i].squeeze()
#     plt.imshow(img, cmap='gray')
#     plt.title(f"Label: {y_train[i]}")
#     plt.axis("off")
#     plt.show()
import numpy as np
import pandas as pd
import os
import cv2
import random
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import threading
from improved_preprocessing import ImagePreprocessor  # Import clasa îmbunătățită

# === CONFIG ===
IMG_TRAIN_PATH = "./dataset/test_v2/test"
IMG_VALID_PATH = "./dataset/validation_v2/validation"
IMG_TEST_PATH = "./dataset/test_v2/test"
CSV_TRAIN_PATH = "./dataset/written_name_test_v2.csv"
CSV_VALID_PATH = "./dataset/written_name_validation_v2.csv"
CSV_TEST_PATH = "./dataset/written_name_test_v2.csv"

# ÎMBUNĂTĂȚIRE: Dimensiune mărită pentru calitate mai bună
IMG_SIZE = (256, 128)  
MIN_SAMPLES_PER_ID = 4
MULTI_WRITER_AUTHORS = 2
MULTI_WRITER_RATIO = 0.5

# Inițializează preprocessor-ul îmbunătățit
preprocessor = ImagePreprocessor(target_size=IMG_SIZE)

# === 1. Load CSVs and merge ===
def load_csv_with_split(path, split_name):
    df = pd.read_csv(path)
    df["split"] = split_name
    return df

df_train = load_csv_with_split(CSV_TRAIN_PATH, "train")
df_val = load_csv_with_split(CSV_VALID_PATH, "val")
df_test = load_csv_with_split(CSV_TEST_PATH, "test")
df = pd.concat([df_train, df_val, df_test], ignore_index=True)
df.dropna(subset=["FILENAME", "IDENTITY"], inplace=True)

# === 2. Resolve full image paths ===
def resolve_path(filename):
    for path in [IMG_TRAIN_PATH, IMG_VALID_PATH, IMG_TEST_PATH]:
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            return full_path
    return None

df["image_path"] = df["FILENAME"].apply(resolve_path)
df = df[df["image_path"].notnull()]

# === 3. Filter authors with enough samples ===
df = df.groupby("IDENTITY").filter(lambda x: len(x) >= MIN_SAMPLES_PER_ID)

# === 4. Label encoding ===
le = LabelEncoder()
df["label_id"] = le.fit_transform(df["IDENTITY"])

# === 7. Group images by author ===
author_to_images = {}
for _, row in df.iterrows():
    author = row['label_id']
    path = row['image_path']
    author_to_images.setdefault(author, []).append(path)

# === 8. Prepare train and val DataFrames ===
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']

def prepare_split_improved(df_split):
    """
    ÎMBUNĂTĂȚIRE: Funcție îmbunătățită pentru prepararea datelor
    """
    authors_in_split = df_split['label_id'].unique()
    split_author_images = {a: author_to_images[a] for a in authors_in_split if a in author_to_images}
    
    X = []
    y = []
    
    print(f"Processing {len(split_author_images)} authors with improved quality...")
    
    # Single-writer samples (label=0)
    processed_count = 0
    for author, paths in split_author_images.items():
        for p in paths:
            try:
                # ÎMBUNĂTĂȚIRE: Folosește preprocessing îmbunătățit
                img = preprocessor.preprocess_image_improved(p)
                img = np.expand_dims(img, axis=-1)
                X.append(img)
                y.append(0)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} single-writer images...")
                    
            except Exception as e:
                print(f"Error processing {p}: {e}")
                continue
    
    # Multi-writer samples (label=1)
    num_multi_samples = int(len(X) * MULTI_WRITER_RATIO)
    authors_list = list(split_author_images.keys())
    
    print(f"Creating {num_multi_samples} multi-writer samples...")
    
    for i in range(num_multi_samples):
        try:
            chosen_authors = random.sample(authors_list, MULTI_WRITER_AUTHORS)
            img_paths = []
            for a in chosen_authors:
                img_path = random.choice(split_author_images[a])
                img_paths.append(img_path)
            
            # ÎMBUNĂTĂȚIRE: Folosește crearea îmbunătățită de imagini composite
            composite = preprocessor.create_composite_improved(img_paths, IMG_SIZE)
            if composite is not None:
                composite = np.expand_dims(composite, axis=-1)
                X.append(composite)
                y.append(1)
                
                if (i + 1) % 50 == 0:
                    print(f"Created {i + 1} multi-writer samples...")
                    
        except Exception as e:
            print(f"Error creating composite {i}: {e}")
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

# Procesare cu threading
X_train, y_train, X_val, y_val = None, None, None, None

def load_train():
    global X_train, y_train
    print("[THREAD] Preparing training data with IMPROVED quality...")
    X_train, y_train = prepare_split_improved(train_df)

def load_val():
    global X_val, y_val
    print("[THREAD] Preparing validation data with IMPROVED quality...")
    X_val, y_val = prepare_split_improved(val_df)

# Start threads
train_thread = threading.Thread(target=load_train)
val_thread = threading.Thread(target=load_val)

train_thread.start()
val_thread.start()

train_thread.join()
val_thread.join()

# === Save to .npz ===
SAVE_PATH = "./preprocessed_data_IMPROVED.npz"
np.savez_compressed(SAVE_PATH, 
                   X_train=X_train, y_train=y_train, 
                   X_val=X_val, y_val=y_val)

print(f"[INFO] Saved IMPROVED preprocessed data to {SAVE_PATH}")
print(f"[INFO] Train set: {X_train.shape}, Labels distribution: {np.bincount(y_train)}")
print(f"[INFO] Val set: {X_val.shape}, Labels distribution: {np.bincount(y_val)}")

# === Visualize improved results ===
print("\n=== VIZUALIZARE REZULTATE ÎMBUNĂTĂȚITE ===")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Imagini cu Calitate Îmbunătățită', fontsize=16)

for i in range(8):
    row = i // 4
    col = i % 4
    
    if i < len(X_train):
        img = X_train[i].squeeze()
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f"Label: {y_train[i]} ({'Single' if y_train[i]==0 else 'Multi'}-writer)")
        axes[row, col].axis("off")
    else:
        axes[row, col].axis("off")

plt.tight_layout()
plt.show()