import os
import shutil
import random


def split_dataset(data_dir, output_dir, train_ratio=0.7, val_ratio=0.15, random_seed=42):
    """
    Splits the dataset into train, val, and test directories.

    Args:
    - data_dir: Path to the folder with all images.
    - output_dir: Path to the output directory where the splits will be saved.
    - train_ratio: Ratio of the data to be used for training.
    - val_ratio: Ratio of the data to be used for validation.
    - random_seed: Seed for random number generator for reproducibility.
    """
    random.seed(random_seed)
    classes = os.listdir(data_dir)

    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue  # Skip non-directory files

        images = os.listdir(class_dir)
        random.shuffle(images)  # Shuffle the images randomly

        # Calculate the split sizes
        total_images = len(images)
        train_size = int(total_images * train_ratio)
        val_size = int(total_images * val_ratio)
        test_size = total_images - train_size - val_size

        # Split the images
        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]

        # Create the necessary output directories
        train_output_dir = os.path.join(output_dir, 'train', class_name)
        val_output_dir = os.path.join(output_dir, 'val', class_name)
        test_output_dir = os.path.join(output_dir, 'test', class_name)

        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(val_output_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)

        # Copy the images to the respective directories
        for img in train_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(train_output_dir, img))

        for img in val_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(val_output_dir, img))

        for img in test_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(test_output_dir, img))

        print(f"Class {class_name}: {train_size} train, {val_size} val, {test_size} test images.")

    print("Dataset splitting completed.")


# Example usage:
data_dir = r"C:\Users\...\Auslan_dataset\dataset_R_G_blur"
output_dir = r"C:\Users\...\Auslan_dataset\dataset_split"
split_dataset(data_dir, output_dir)
