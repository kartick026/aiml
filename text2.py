import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
dataset_path = "data.csv"  # Update this path if needed
df = pd.read_csv(dataset_path)

# Display the first few rows of the dataset
print("Dataset Head:")
print(df.head())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Extract labels and images
labels = df.iloc[:, 0]  # First column contains labels
images = df.iloc[:, 1:].values  # Remaining columns contain pixel values

# Reshape images (28x28)
images = images.reshape(-1, 28, 28)

# Display sample images from each category
unique_labels = np.unique(labels)

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.ravel()

for i, label in enumerate(unique_labels[:10]):  # Display first 10 categories
    index = np.where(labels == label)[0][0]  # Find first occurrence of each label
    axes[i].imshow(images[index], cmap='gray')
    axes[i].set_title(f"Label: {label}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
