"""
Jesse Han
jesse.han53@myhunter.cuny.edu
CSCI 39534 Lab 7
Resources: Gemini for numpy FFT resources
"""

from PIL import Image
import numpy as np
import math

# Sourced from F_Fourier_Image_Low_Pass_Examples.m
def low_pass(image, D0=30, n=2):
    cols = image.size[0]
    rows = image.size[1]
    img = image.copy()
    pxl = img.load()

    img_array = np.array(img)

    # Fourier Transform of the Image
    fft = np.fft.fft2(img_array)
    fft_shifted = np.fft.fftshift(fft)

    ideal_mask = np.zeros((rows, cols), int)
    gaussian_mask = np.zeros((rows, cols), int)
    butterworth_mask = np.zeros((rows, cols), int)

    for i in range(cols):
        for j in range(rows):
            D = np.sqrt((i-(cols//2))**2 + (j - (rows//2))**2)
            if D <= D0:
                ideal_mask[i,j] = 1
                gaussian_mask[i,j] = math.exp(-(float(D) ** 2.0) / (2.0 * float(D0) ** 2.0))
                butterworth_mask[i,j] = 1.0 / (1.0 + (float(D) / float(D0)) ** (2.0 * float(n)))

    ideal = fft_shifted * ideal_mask
    gaussian = fft_shifted * gaussian_mask
    butterworth = fft_shifted * butterworth_mask

    ideal_img = np.real(np.fft.ifft2(np.fft.ifftshift(ideal)))
    ideal_img = ((ideal_img - np.min(ideal_img)) / (np.max(ideal_img) - np.min(ideal_img))) * 255
    ideal_img = Image.fromarray(np.uint8(ideal_img))
    
    gaussian_img = np.real(np.fft.ifft2(np.fft.ifftshift(gaussian)))
    gaussian_img = ((gaussian_img - np.min(gaussian_img)) / (np.max(gaussian_img) - np.min(gaussian_img))) * 255
    gaussian_img = Image.fromarray(np.uint8(gaussian_img))
    
    butterworth_img = np.real(np.fft.ifft2(np.fft.ifftshift(butterworth)))

    ideal_img.save('ideal_dog.png')
    gaussian_img.save('gaussian_dog.png')
    #butterworth_img.save('butterworth_dog.png')


dog = Image.open('dog.png').convert('L')
dog.save('grayscale_dog.png')

lowpass_dog = low_pass(dog)

