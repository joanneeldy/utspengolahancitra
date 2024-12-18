"""
UTS Pengolahan Citra
Joanne Landy Tantreece
NIM: 210401010022
Dosen: Alun Sujjada, S.Kom., M.T

Meningkatkan kontras citra menggunakan histogram equalization
& pengaturan level kontras tertentu.
Berikut adalah hasil coding program untuk menjawab soal nomor 2 dan 3.

Terima kasih
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# (2) Melakukan histogram equalization
def histogram_equalization(input_image):
    """
    Meningkatkan kontras citra dengan histogram equalization.
    Parameter:
    input_image : numpy.array (Citra yang akan diperbaiki kontrasnya)
    Return:
    numpy.array (Citra yang telah diperbaiki kontrasnya)
    """
    hist, _ = np.histogram(input_image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()  # Cumulative Distribution Function

    # Normalisasi CDF
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # Aplikasi transformasi ke citra
    img_eq = cdf[input_image]
    return img_eq

# (3) Kontras level tertentu
def adjust_contrast(input_image, contrast_level):
    """
    Meningkatkan kontras citra dengan mengatur level kontrasnya.
    Parameter:
    input_image : numpy.array (Citra yang akan diperbaiki kontrasnya)
    contrast_level : int (Nilai level kontras yang diinginkan, 0-254)
    Return:
    numpy.array (Citra yang telah diperbaiki kontrasnya)
    """
    factor = (259 * (contrast_level + 255)) / (255 * (259 - contrast_level))
    adjusted_image = 128 + factor * (input_image - 128)
    return np.clip(adjusted_image, 0, 255).astype('uint8')

# Memuat citra kontras rendah
IMAGE_PATH = "ori_image.jpg"

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"File gambar '{IMAGE_PATH}' tidak ditemukan!")

original_image = imageio.imread(IMAGE_PATH, pilmode="L")
if original_image.max() <= 1:  # Normalisasi jika gambar berbentuk float (0-1)
    original_image = (original_image * 255).astype('uint8')

# Histogram Equalization
equalized_image = histogram_equalization(original_image)

# Peningkatan kontras level 1.5
CONTRAST_LEVEL = 1.5
contrast_adjusted_image = adjust_contrast(original_image, CONTRAST_LEVEL)

# Visualisasi hasil
plt.figure(figsize=(14, 10))

# Citra Asli
plt.subplot(3, 2, 1)
plt.title("Citra Asli")
plt.imshow(original_image, cmap='gray')
plt.axis('off')

plt.subplot(3, 2, 2)
plt.title("Histogram Citra Asli")
plt.hist(original_image.flatten(), bins=256, color='blue', alpha=0.7)
plt.xlabel("Intensitas")
plt.ylabel("Frekuensi")

# Histogram Equalization
plt.subplot(3, 2, 3)
plt.title("Histogram Equalization")
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

plt.subplot(3, 2, 4)
plt.title("Histogram Hasil Equalization")
plt.hist(equalized_image.flatten(), bins=256, color='green', alpha=0.7)
plt.xlabel("Intensitas")
plt.ylabel("Frekuensi")

# Penyesuaian Level Kontras
plt.subplot(3, 2, 5)
plt.title(f"Contrast Adjustment (Level {CONTRAST_LEVEL})")
plt.imshow(contrast_adjusted_image, cmap='gray')
plt.axis('off')

plt.subplot(3, 2, 6)
plt.title("Histogram Hasil Kontras Adjustment")
plt.hist(contrast_adjusted_image.flatten(), bins=256, color='red', alpha=0.7)
plt.xlabel("Intensitas")
plt.ylabel("Frekuensi")

plt.tight_layout()
plt.show()

# Menyimpan citra hasil
imageio.imwrite("equalized_image.jpg", equalized_image)
imageio.imwrite("contrast_adjusted_image.jpg", contrast_adjusted_image)

print("Citra yang telah diperbaiki berhasil disimpan sebagai:")
print("1. 'equalized_image.jpg' -> Hasil Histogram Equalization")
print("2. 'contrast_adjusted_image.jpg' -> Hasil Pengaturan Kontras")
