#%%
import os
import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import matplotlib.pyplot as plt

def load_images_by_subdir(base_dir, size=(256, 256)):
    grouped_images = defaultdict(list)
    for root, dirs, files in os.walk(base_dir):
        for file in sorted(files):
            if file.lower().endswith(('.png')):
                filepath = os.path.join(root, file)
                rel_subdir = os.path.relpath(root, base_dir)
                img = Image.open(filepath).convert('RGB').resize(size)
                grouped_images[rel_subdir].append((file, np.asarray(img)))
    return grouped_images

def compute_mse_matrix(images):
    n = len(images)
    mse_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i <= j:
                mse = mean_squared_error(images[i][1].flatten(), images[j][1].flatten())
                mse_matrix[i, j] = mse
                #mse_matrix[j, i] = mse
    return mse_matrix

def save_mse_heatmap(matrix, output_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='viridis')
    plt.title('MSE Error Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == '__main__':

    base_directory = 'D:\\DAS\\FK\\NoShore_DS3_x512_20250604T143445\\FK'
    grouped = load_images_by_subdir(base_directory)

    for subdir, images in grouped.items():
        print(subdir)
        if len(images) < 2:
            continue
        mse_matrix = compute_mse_matrix(images)
        image_names = [img[0] for img in images]
        
        output_file = os.path.join(base_directory, subdir, "mse_matrix.jpg")
        save_mse_heatmap(mse_matrix, output_file)

# %%
