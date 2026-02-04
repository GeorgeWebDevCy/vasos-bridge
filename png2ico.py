from PIL import Image
import os

img = Image.open("cucumber.png")
img.save("cucumber.ico", format="ICO", sizes=[(256, 256)])
print("Converted cucumber.png to cucumber.ico")
