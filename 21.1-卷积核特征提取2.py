import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

image = np.zeros((100, 100))
image[30:70, 30:70] = 255  # 白色方块
image = image + np.random.normal(0, 20, image.shape)  # 加点噪声 0表示mean， 20为std
image = np.clip(image, 0, 255)
# 定义一个3x3平均模糊卷积核
blur_kernel = np.ones((3, 3)) / 9.0

# 对图像进行卷积（模糊处理）
blur_image = ndimage.convolve(image, blur_kernel)

# 增加池化操作（3x3平均池化）
pool_kernel_size = 3
pooled_image = ndimage.uniform_filter(blur_image, size=pool_kernel_size)

# 显示原图、卷积后和池化后的结果
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.title("原始图像")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("卷积后 (模糊效果)")
plt.imshow(blur_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("池化后 (降采样)")
plt.imshow(pooled_image, cmap="gray")
plt.axis("off")

plt.show()
