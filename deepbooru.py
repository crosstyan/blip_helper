import os.path
import re
import zipfile
import deepdanbooru as dd
import tensorflow as tf
import numpy as np

# hate it
# TODO: use another downloader
from basicsr.utils.download_util import load_file_from_url

pwd = os.path.dirname(os.path.realpath(__file__))
default_deepbooru_model_path = os.path.abspath(os.path.join(pwd, "pretrained", "deepbooru"))

re_special = re.compile(r"([\\()])")

def get_deepbooru_tags_model(model_path: str):
    # why do you find DeepBooru in the fucking temp by default?
    if not os.path.exists(os.path.join(model_path, "project.json")):
        is_abs = os.path.isabs(model_path)
        if not is_abs:
            model_path = os.path.abspath(model_path)
        # there is no point importing these every time
        load_file_from_url(
            r"https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/deepdanbooru-v3-20211112-sgd-e28.zip",
            model_path,
        )
        with zipfile.ZipFile(
            os.path.join(model_path, "deepdanbooru-v3-20211112-sgd-e28.zip"), "r"
        ) as zip_ref:
            zip_ref.extractall(model_path)
        os.remove(os.path.join(model_path, "deepdanbooru-v3-20211112-sgd-e28.zip"))

    tags = dd.project.load_tags_from_project(model_path)
    model = dd.project.load_model_from_project(model_path, compile_model=False)
    return model, tags


def get_deepbooru_tags_from_model(
    model,
    tags,
    pil_image,
    threshold,
    alpha_sort=False,
    use_spaces=True,
    use_escape=True,
    include_ranks=False,
):
    width = model.input_shape[2]
    height = model.input_shape[1]
    image = np.array(pil_image)
    image = tf.image.resize(
        image,
        size=(height, width),
        method=tf.image.ResizeMethod.AREA,
        preserve_aspect_ratio=True,
    )
    image = image.numpy()  # EagerTensor to np.array
    image = dd.image.transform_and_pad_image(image, width, height)
    image = image / 255.0
    image_shape = image.shape
    image = image.reshape((1, image_shape[0], image_shape[1], image_shape[2]))

    y = model.predict(image)[0]

    result_dict = {}

    for i, tag in enumerate(tags):
        result_dict[tag] = y[i]

    unsorted_tags_in_theshold = []
    result_tags_print = []
    for tag in tags:
        if result_dict[tag] >= threshold:
            if tag.startswith("rating:"):
                continue
            unsorted_tags_in_theshold.append((result_dict[tag], tag))
            result_tags_print.append(f"{result_dict[tag]} {tag}")

    # sort tags
    result_tags_out = []
    sort_ndx = 0
    if alpha_sort:
        sort_ndx = 1

    # sort by reverse by likelihood and normal for alpha, and format tag text as requested
    unsorted_tags_in_theshold.sort(key=lambda y: y[sort_ndx], reverse=(not alpha_sort))
    for weight, tag in unsorted_tags_in_theshold:
        tag_outformat = tag
        if use_spaces:
            tag_outformat = tag_outformat.replace("_", " ")
        if use_escape:
            tag_outformat = re.sub(re_special, r"\\\1", tag_outformat)
        if include_ranks:
            tag_outformat = f"({tag_outformat}:{weight:.3f})"

        result_tags_out.append(tag_outformat)

    print("\n".join(sorted(result_tags_print, reverse=True)))

    return ", ".join(result_tags_out)