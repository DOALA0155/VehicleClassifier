from PIL import Image
import os
import numpy as np

def get_image_data():
    vehicles = ["car", "train", "ship", "airplane", "bicycle"]
    images = []
    labels = []

    for index, vehicle in enumerate(vehicles):
        dir_path = "./images/{}".format(vehicle)
        image_paths = os.listdir(dir_path)

        for j, image_path in enumerate(image_paths):
            image = np.asarray(Image.open(dir_path + "/" + image_path))
            if image.shape == (150, 150):
                continue
            images.append(image)
            labels.append(index)

    images = np.asarray(images)
    labels = np.asarray(labels)
    return images, labels
