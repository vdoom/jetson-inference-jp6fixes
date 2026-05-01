#!/usr/bin/env python3
#
# Test script for Cyrillic text rendering with cudaFont.
# Renders Latin and Cyrillic text onto an image and saves it.
#

from jetson_utils import cudaAllocMapped, cudaFont, saveImage, cudaDeviceSynchronize

# create a blank image (dark blue background)
width, height = 800, 900
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

# --- background color variations ---

# fully transparent background (no fill, text on raw image)
font.OverlayText(img, text="Transparent bg / Прозорий фон",
                 x=20, y=410,
                 color=font.White, background=(0, 0, 0, 0))

# solid black background
font.OverlayText(img, text="Black bg / Чорний фон",
                 x=20, y=460,
                 color=font.White, background=(0, 0, 0, 255))

# solid red background, white text
font.OverlayText(img, text="Red bg / Червоний фон",
                 x=20, y=510,
                 color=font.White, background=(200, 30, 30, 255))

# solid green background, black text
font.OverlayText(img, text="Green bg / Зелений фон",
                 x=20, y=560,
                 color=font.Black, background=(40, 180, 60, 255))

# solid blue background, yellow text
font.OverlayText(img, text="Blue bg / Синій фон",
                 x=20, y=610,
                 color=(255, 240, 80, 255), background=(40, 80, 200, 255))

# semi-transparent white background (alpha 128)
font.OverlayText(img, text="Semi-white bg / Напівпрозорий білий",
                 x=20, y=660,
                 color=font.Black, background=(255, 255, 255, 128))

# semi-transparent black background (alpha 100)
font.OverlayText(img, text="Semi-black bg / Напівпрозорий чорний",
                 x=20, y=710,
                 color=font.White, background=(0, 0, 0, 100))

# solid magenta background, cyan text
font.OverlayText(img, text="Magenta+Cyan / Пурпурний+Блакитний",
                 x=20, y=760,
                 color=(80, 240, 255, 255), background=(220, 40, 200, 255))

# solid orange background, dark text
font.OverlayText(img, text="iiiiiiiiiii",#"Orange bg / Помаранчевий фон",
                 x=20, y=810,
                 color=(40, 20, 0, 255), background=(255, 150, 40, 255))

cudaDeviceSynchronize()

# save the result
output_path = "test_cyrillic.jpg"
saveImage(output_path, img)
print(f"saved test image to {output_path}")
