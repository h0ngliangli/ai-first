import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# 生成一张简单的灰度图像 (矩形 + 噪声)
image = np.zeros((100, 100))
image[30:70, 30:70] = 255  # 白色方块
image = image + np.random.normal(0, 20, image.shape)  # 加点噪声
image = np.clip(image, 0, 255)

# 定义一个3x3边缘检测卷积核（Sobel-like for vertical edges）
kernel = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

# 对图像进行卷积
edge_image = ndimage.convolve(image, kernel)

# 显示原图与卷积后的结果
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("原始图像")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("卷积后 (检测垂直边缘)")
plt.imshow(edge_image, cmap='gray')
plt.axis('off')

plt.show()
