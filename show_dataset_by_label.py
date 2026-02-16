import numpy as np
import matplotlib.pyplot as plt

# Încarcă datele
data = np.load("./preprocessed_data.npz")
X_train = data["X_train"]
y_train = data["y_train"]


# Găsește indexii imaginilor "multi-writer" (eticheta 1)
multi_writer_indices = np.where(y_train == 0)[0]

# Afișează primele 8 imagini "multi-writer"
num_images = 30
fig, axes = plt.subplots(2, 4, figsize=(10, 5))

for idx, ax in enumerate(axes.flat):
    if idx < len(multi_writer_indices):
        i = multi_writer_indices[idx]
        ax.imshow(X_train[i].squeeze(), cmap="gray")
        ax.set_title("Multi-writer")
        ax.axis("off")
    else:
        ax.axis("off")

plt.tight_layout()
plt.show()

# import numpy as np
# print(np.unique(y_val, return_counts=True))
