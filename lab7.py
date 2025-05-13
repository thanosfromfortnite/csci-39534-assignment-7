"""
Jesse Han
jesse.han53@myhunter.cuny.edu
CSCI 39534 Lab 7
Resources: Gemini for numpy FFT resources
"""

from PIL import Image
import numpy as np

def ideal_low_pass(image, cutoff=30):
    img = image.copy()

    # Image -> array for np manipulation
    img_array = np.array(img)
    
    f_transform = np.fft.fft2(img_array)
    f_shift = np.fft.fftshift(f_transform)

    rows, cols = img_array.shape
    mask = np.zeros((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - (rows // 2))**2 + (j - (cols // 2))**2) <= cutoff:
                mask[i, j] = 1

    f_shift_filtered = f_shift * mask

    f_inverse_shift = np.fft.ifftshift(f_shift_filtered)
    img_filtered = np.fft.ifft2(f_inverse_shift)
    img_filtered = np.real(img_filtered)

    img_filtered = (img_filtered - np.min(img_filtered)) / (np.max(img_filtered) - np.min(img_filtered)) * 255
    img_filtered = np.uint8(img_filtered)
    
    return Image.fromarray(img_filtered)

dog = Image.open('dog.png').convert('L')
dog.save('grayscale_dog.png')

lowpass_dog = ideal_low_pass(dog)
lowpass_dog.save('lowpass_dog.png')

