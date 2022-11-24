import os.path
import zipfile
from pathlib import Path
from download_util import load_file_from_url

pwd = os.path.dirname(os.path.realpath(__file__))
default_deepbooru_model_path = Path(os.path.abspath(
    os.path.join(pwd, "pretrained", "deepbooru")))


def init_deepbooru():
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        # prevent tensorflow from using all the VRAM
        tf.config.experimental.set_memory_growth(gpu, True)
    model, tags = get_deepbooru_tags_model(default_deepbooru_model_path)
    return model, tags

# TODO: refactor this to let user specify the model path


def get_deepbooru_tags_model(dd_path: str):
    """Loads the deepbooru model and returns the model and the tags.

    Args:
        dd_path (str): DeepDanbooru pretrained model path.

    Returns:
        model_path, tags: The model and the tags.
        tuple[Path, list[str]]
    """
    base_path = Path(dd_path)
    if not (base_path / "project.json").exists():
        is_abs = os.path.isabs(dd_path)
        if not is_abs:
            dd_path = os.path.abspath(dd_path)
        # there is no point importing these every time
        load_file_from_url(
            r"https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/deepdanbooru-v3-20211112-sgd-e28.zip",
            dd_path,
        )
        with zipfile.ZipFile(
            os.path.join(
                dd_path, "deepdanbooru-v3-20211112-sgd-e28.zip"), "r"
        ) as zip_ref:
            zip_ref.extractall(dd_path)
        os.remove(os.path.join(
            dd_path, "deepdanbooru-v3-20211112-sgd-e28.zip"))

    tags_path = base_path / "tags.txt"
    model_path = base_path / "model-resnet_custom_v3.h5"
    tags = []
    with open(tags_path, "r") as f:
        tags = f.read().splitlines()
    return model_path, tags
