import numpy as np
import struct
import os
from PIL import Image

def parse_idx3_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError("Invalid IDX3 image file")
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def parse_idx3_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError("Invalid IDX3 label file")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def save_images(images, labels, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, (image, label) in enumerate(zip(images, labels)):
        img = Image.fromarray(image)
        img.save(os.path.join(output_dir, f'{label}_{i}.png'))
        if i % 100 == 0:
            break

if __name__ == "__main__":
    train_images_path = 'data/FashionMNIST/raw/train-images-idx3-ubyte'
    train_labels_path = 'data/FashionMNIST/raw/train-labels-idx1-ubyte'
    test_images_path = 'data/FashionMNIST/raw/t10k-images-idx3-ubyte'
    test_labels_path = 'data/FashionMNIST/raw/t10k-labels-idx1-ubyte'

    train_images = parse_idx3_images(train_images_path)
    train_labels = parse_idx3_labels(train_labels_path)
    test_images = parse_idx3_images(test_images_path)
    test_labels = parse_idx3_labels(test_labels_path)

    save_images(train_images, train_labels, 'data/FashionMNIST/raw/train_images')
    save_images(test_images, test_labels, 'data/FashionMNIST/raw/test_images')

    print(f'Training images: {len(train_images)}, Training labels: {len(train_labels)}')
    print(f'Test images: {len(test_images)}, Test labels: {len(test_labels)}')
    print('Images saved to data/FashionMNIST/raw/train_images and data/FashionMNIST/raw/test_images')