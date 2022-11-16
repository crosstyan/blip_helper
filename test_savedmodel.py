import os
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate

from Utils import dbimutils

pd.set_option("display.max_rows", 1000)

use_GPU = False
if use_GPU == False:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from tensorflow.keras.models import load_model

image_size = 448
thresh = 0.35

model = load_model("networks/ViTB16_11_03_2022_07h05m53s")
label_names = pd.read_csv("2022_0000_0899_6549/selected_tags.csv")

target_img = "82148729_p0.jpg" if len(sys.argv) < 2 else sys.argv[1]

img = dbimutils.smart_imread(target_img)
img = dbimutils.smart_24bit(img)
img = dbimutils.make_square(img, image_size)
img = dbimutils.smart_resize(img, image_size)
img = img.astype(np.float32)
img = np.expand_dims(img, 0)

probs = model(img, training=False)

label_names["probs"] = probs[0]

# First 4 labels are actually ratings: pick one with argmax
ratings_names = label_names[:4]
rating_index = ratings_names["probs"].argmax()
found_rating = ratings_names[rating_index : rating_index + 1][["name", "probs"]]

# Everything else is tags: pick any where prediction confidence > threshold
tags_names = label_names[4:]
found_tags = tags_names[tags_names["probs"] > thresh][["tag_id", "name", "probs"]]

print(tabulate(found_rating, headers="keys"))
print()
print(tabulate(found_tags, headers="keys"))
