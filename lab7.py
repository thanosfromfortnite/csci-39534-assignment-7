"""
Jesse Han
jesse.han53@myhunter.cuny.edu
CSCI 39534 Lab 7
Resources: Gemini for numpy FFT resources
"""

from PIL import Image
import numpy as np

def ideal_low_pass(image, cutoff=30):
    width = image.size[0]
    height = image.size[1]
    img = image.copy()
    pxl = img.load()

    img_array = np.array(img)

    fft = np.fft.fft2(img_array)
    fft_shifted = np.fft.fftshift(fft)

    for i in range(width):
        for j in range(height):
            if np.sqrt() <= cutoff

dog = Image.open('dog.png').convert('L')
dog.save('grayscale_dog.png')

lowpass_dog = ideal_low_pass(dog)

