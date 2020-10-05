import os

images = os.listdir('all_images(val)/images')

with open('all_images(val)/test.txt', 'w') as f:
    for image in images:
        image = image.split('.')[0]
        f.write(image + '\n')
