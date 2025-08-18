import struct
import numpy as np
from PIL import Image

label_to_str = {
    0: "Top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

labels = []


with open('data/FashionMNIST/raw/train-labels-idx1-ubyte', 'rb') as f:
    magic, num_labels = struct.unpack('>II', f.read(8))
    print(f'Magic: {magic}, Number of labels: {num_labels}')
    labels = f.read(num_labels)
    print(f'Total labels read: {len(labels)}')
    print(f'Labels: {[label_to_str[label] for label in labels[:10]]}...')  # Print first 10 labels for brevity

def export_images(file_path, dest):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        print(f'Magic: {magic}, Number of images: {num_images}, Rows: {rows}, Columns: {cols}')
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        print(f'Total images read: {len(images)}, Shape: {images.shape}')
        for i in range(10):
            file_name = f'{dest}/{i}_{label_to_str[labels[i]]}.png'
            print(f'Saving image {i} to {file_name}')
            img = Image.fromarray(images[i])
            img.save(file_name)


export_images('data/FashionMNIST/raw/train-images-idx3-ubyte', 'data/FashionMNIST/raw/train_images')
export_images('data/FashionMNIST/raw/t10k-images-idx3-ubyte', 'data/FashionMNIST/raw/test_images')