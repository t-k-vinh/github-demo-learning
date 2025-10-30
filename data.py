import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Tải và lọc dữ liệu ---

# Tải bộ dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Gộp dữ liệu train và test lại
x_all = np.concatenate((x_train, x_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)

# Lọc ra chỉ số 3 và số 7
mask = (y_all == 3) | (y_all == 7)
x_filtered = x_all[mask]
y_filtered = y_all[mask]

# Lấy 1000 ảnh cho mỗi loại
x_3s = x_filtered[y_filtered == 3][:1000]
x_7s = x_filtered[y_filtered == 7][:1000]

print(f"Đã tải xong 1000 ảnh số 3, kích thước: {x_3s.shape}")
print(f"Đã tải xong 1000 ảnh số 7, kích thước: {x_7s.shape}")

# --- 2. Hiển thị 10 số 3 đầu tiên ---

print("\nĐang hiển thị 10 số 3 đầu tiên...")
plt.figure(figsize=(10, 4)) # Đặt kích thước cho cửa sổ hình ảnh
plt.suptitle("10 ảnh 'số 3' đầu tiên", fontsize=16)

for i in range(10):
    # Tạo một ô con trong lưới 2x5 (2 hàng, 5 cột)
    plt.subplot(2, 5, i + 1)
    
    # Hiển thị ảnh (dùng cmap='gray' cho ảnh đen trắng)
    plt.imshow(x_3s[i], cmap='gray')
    
    # Tắt các trục tọa độ
    plt.axis('off')

# --- 3. Hiển thị 10 số 7 đầu tiên ---

print("Đang hiển thị 10 số 7 đầu tiên...")
plt.figure(figsize=(10, 4)) # Tạo một cửa sổ hình ảnh mới
plt.suptitle("10 ảnh 'số 7' đầu tiên", fontsize=16)

for i in range(10):
    # Tạo một ô con trong lưới 2x5
    plt.subplot(2, 5, i + 1)
    
    # Hiển thị ảnh
    plt.imshow(x_7s[i], cmap='gray')
    
    # Tắt các trục tọa độ
    plt.axis('off')

# Hiển thị cả hai cửa sổ hình ảnh
plt.show()