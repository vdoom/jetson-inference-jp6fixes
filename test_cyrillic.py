#!/usr/bin/env python3
#
# Test script for Cyrillic text rendering with cudaFont.
# Renders Latin and Cyrillic text onto an image and saves it.
#

from jetson_utils import cudaAllocMapped, cudaFont, saveImage, cudaDeviceSynchronize

# create a blank image (dark blue background)
width, height = 800, 600
img = cudaAllocMapped(width=width, height=height, format='rgb8')

# fill with a dark background color
for y in range(height):
    for x in range(width):
        img[y, x] = (30, 30, 60)

cudaDeviceSynchronize()

# create font
font = cudaFont(size=32)

# render Latin text
font.OverlayText(img, text="Hello, World! 0123456789",
                 x=20, y=20,
                 color=font.White, background=font.Gray40)

# render Ukrainian text
font.OverlayText(img, text="Привіт, Світ! Тест кирилиці.",
                 x=20, y=80,
                 color=(120, 255, 120, 255), background=font.Gray40)

# render mixed Latin + Ukrainian
font.OverlayText(img, text="Mixed: Jetson Inference + Кирилиця",
                 x=20, y=140,
                 color=(120, 200, 255, 255), background=font.Gray40)

# render Ukrainian-specific letters
font.OverlayText(img, text="Українська: Ґ ґ Є є І і Ї ї",
                 x=20, y=200,
                 color=(255, 255, 100, 255), background=font.Gray40)

# render the full Ukrainian alphabet
font.OverlayText(img, text="АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ",
                 x=20, y=280,
                 color=font.White, background=font.Gray40)

font.OverlayText(img, text="абвгґдеєжзиіїйклмнопрстуфхцчшщьюя",
                 x=20, y=340,
                 color=font.White, background=font.Gray40)

cudaDeviceSynchronize()

# save the result
output_path = "test_cyrillic.jpg"
saveImage(output_path, img)
print(f"saved test image to {output_path}")
