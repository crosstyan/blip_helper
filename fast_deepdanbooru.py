# code from the one and only AO
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import argparse
from multiprocessing import Pool, Queue, Process, Manager
from typing import List
import skimage
import json
from PIL import Image
import multiprocessing as mp
import re
from typing import List, Tuple, Generator
import numpy as np


def _run_dd_keras_prediction(img_gen: Generator[np.ndarray, None, None], send_end, h5_path: str, batch_size: int):
    """Condom wrapper for keras prediction. This shit doesn't free vram once it's loaded

    Args:
        img_gen (Generator): yields (idx, img_array)
        send_end (pipe): pipe to send result
        batch_size (int): Defaults to 8.
        h5_path (str): path to the pre-trained model.

    Returns:
        None. Result is sent through pipe
    """

    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert gpus, "No GPU found"
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    idx_order = []

    def dummy_gen():
        for i, dat in img_gen:
            idx_order.append(i)
            yield dat
    ds = tf.data.Dataset.from_generator(dummy_gen, output_types=tf.float32, output_shapes=(512, 512, 3))
    ds = ds.batch(batch_size)
    print("Loading model...")
    model: tf.keras.Model = tf.keras.models.load_model(h5_path)  # type: ignore
    print("Model loaded. Starting prediction...")
    r = model.predict(ds, batch_size=batch_size)
    print("Prediction done. Results sent")
    send_end.send((idx_order, r))


def run_dd_keras_prediction(img_gen: Generator[np.ndarray, None, None], model_path: Path, batch_size: int = 8):
    """ Runs DD keras prediction in a separate process to prevent VRAM leak.
    Args:
        img_gen (Generator): yields (idx, img_array)
        batch_size (int): Defaults to 8.
        model_path (str): path to the pre-trained model.

    Returns:
        Tuple[List[int], np.ndarray]: idx_order, predictions
    """
    recv_end, send_end = mp.Pipe(False)
    p = mp.Process(target=_run_dd_keras_prediction, args=(img_gen, send_end, model_path, batch_size))
    p.start()
    r = recv_end.recv()
    p.join()
    return r


def transform_and_pad_image(image, target_width, target_height, order=3, mode="edge"):
    """
    Transform image and pad by edge pixles.
    Args:
        image (PIL.Image.Image): image to transform
        target_width (int): target width
        target_height (int): target height
        order (int): order of interpolation. 0 for nearest neighbor, 1 for bilinear interpolation, 3 for bicubic interpolation.
        mode (str): mode of padding. Defaults to "edge".
    Returns:
        np.ndarray: transformed image    
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    w, h = image.size
    np_image = np.array(image)
    if np_image.shape[-1] == 4:
        np_image[:, :, :3] = (np_image[:, :, 3:] / 255 * np_image[:, :, :3] + (1 - np_image[:, :, 3:] / 255) * 255).astype(np.uint8)
        np_image = np_image[:, :, :3]

    if w == target_width and h == target_height:
        return np.array(np_image / np.float32(255), dtype=np.float32)

    t = skimage.transform.AffineTransform(
        translation=((target_width - w) * 0.5, (target_height - h) * 0.5)
    )

    warp_shape = (target_height, target_width)

    image_array = skimage.transform.warp(
        np_image, (t).inverse, output_shape=warp_shape, order=order, mode=mode
    )

    return image_array.astype(np.float32)


def dd_mt_gen_worker(x):
    """Worker function for dd_mt_gen. This function is called by multiprocessing.Pool.map_async"""
    q, i, img = x
    q.put((i, transform_and_pad_image(img, 512, 512, order=1, mode="edge")))


def dd_mt_gen(images, nproc, maximum_look_ahead: int = 128, x: int = 512, y: int = 512):
    """Multi-threaded image dataloader for deepdanbooru. All necessary preprocesses are done in this function.

    Args:
        image_paths (List[str]): list of image paths or Generator[PIL.Image.Image]
        nproc (int): number of processes. Defaults to 8.
        maximum_look_ahead (int): maximum number of images preload. Defaults to 128.
        x,y (int): width and height of the image to feed into the model. Defaults to 512. Change this if you know what you're doing.

    Yields:
        idx, img_array
    """
    images = list(images) if isinstance(images, Generator) else images
    pool = Pool(nproc)
    manager = Manager()
    q = manager.Queue(maxsize=maximum_look_ahead)

    def gen(q, imgs):
        for i, img in enumerate(imgs):
            yield q, i, img

    r = pool.map_async(dd_mt_gen_worker, gen(q, images))

    for _ in range(len(images)):
        yield q.get()


def get_tag_mask(DD_path, purges: Tuple[str, ...] = ('System',)):
    """
    Set mask to False for tags that are in blacklist categories
    Args:
        DD_path (str): path to folder containing categories.json
        purges: list of category names to purge. All these names should be in cate_names
    """
    with open(Path(DD_path) / 'categories.json') as f:
        cate_list = json.load(f)
    cate_list = sorted(cate_list, key=lambda k: k['start_index'])
    cate_names = [cate['name'] for cate in cate_list]
    cate_idx = [cate['start_index'] for cate in cate_list]
    assert len(cate_idx) == len(cate_names), 'cate_idx and cate_names should have the same length'
    assert all([purge in cate_names for purge in purges]), 'purges should be in cate_names'
    cate_idx = cate_idx + [None]
    mask = np.ones((len(tags),), dtype=bool)
    left = 0
    for name, right in zip(cate_names, cate_idx[1:]):
        if name in purges:
            mask[left:right] = False
        left = right
    return mask


def _dirwalk(path: Path):
    for fname in path.iterdir():
        if fname.is_dir():
            yield from _dirwalk(fname)
        else:
            yield fname


def dirwalk(path: Path):
    """Walks through a directory and yields all file paths in the directory and subdirectories"""
    for fname in _dirwalk(path):
        yield fname


def fix_order(r, order):
    """Fixes the order of the predictions. This is necessary because the predictions are done in parallel and the order is not guaranteed."""
    r2 = r.copy()
    for i, order in enumerate(order):
        r2[order] = r[i]
    return r2


def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', '-i', type=str, required=True, help='Input folder')
    parser.add_argument('--append-mode', '-m', type=str, default='append', choices=['prepend', 'append', 'overwrite', 'skip'])
    parser.add_argument('--DD-path', '-D', type=str, required=True, help='Path to DeepDanbooru folder')
    parser.add_argument('--no-alpha-sort', action='store_true', help='Sort by probability instead of alpha')
    parser.add_argument('--threshold', '-t', type=float, default=0.5)
    parser.add_argument('--no-replace-underscore', action='store_true')
    parser.add_argument('--no-escape-special-char', action='store_true')
    parser.add_argument('--nproc', '-j', type=int, default=-1, help='Number of processes in processing module. Defaults to number of CPU cores')
    parser.add_argument('--batch-size', '-bs', type=int, default=8)
    parser.add_argument('--maximum-look-ahead', type=int, default=128, help='Maximum number of images to preload')
    parser.add_argument('--tag-characters', action='store_true', help='Tag character names (Why)')
    parser.add_argument('--blacklist-tags', nargs='+', type=str, help='Images with these tags will be purged')
    parser.add_argument('--blacklist-threshold', nargs='+', type=float, help='List of thresholds to blacklist')
    parser.add_argument('--purge-tomoe-mami', '-mami', action='store_true', help='Purge Tomoe Mami (adj.) pictures')
    parser.add_argument('--max-number-tags', '-N', type=int, default=30, help='Maximum number of tags to generate')
    parser.add_argument('--purge-text', '-text', action='store_true', help='Purge images with text in them')
    args = parser.parse_args()
    assert args.threshold > 0 and args.threshold <= 1, 'Threshold should be between 0 and 1'
    assert args.maximum_look_ahead > 0, 'Maximum look ahead should be greater than 0'
    assert args.batch_size > 0, 'Batch size should be greater than 0'
    if args.blacklist_tags is not None:
        assert len(args.blacklist_tags) == len(
            args.blacklist_threshold), f'blacklist_tags and blacklist_threshold should have the same length. Got {args.blacklist_tags} and {args.blacklist_threshold}'
        assert all([t > 0 and t <= 1 for t in args.blacklist_threshold]), 'blacklist_threshold should be between 0 and 1'

    print(f'WARNING: using blacklist={args.blacklist_tags} with threshold={args.blacklist_threshold}')
    return args


def purger_gen(tags, blacklist_tags, thres):
    assert len(thres) == len(blacklist_tags)
    tags = list(tags)
    blacklist_index = [tags.index(tag) for tag in blacklist_tags]
    blacklist_index = np.array(blacklist_index)
    print(blacklist_index)
    thres = np.array(thres)

    def purger(prediction, path) -> bool:
        """Returns True if the image was purged"""
        if np.any(prediction[blacklist_index] > thres):
            os.makedirs(path.parent / 'purger', exist_ok=True)
            os.rename(path, path.parent / 'purger' / path.name)
            return True
        return False

    return purger


def mani_detector_gen(tags):
    idx_eyes = [i for i, tag in enumerate(tags) if 'eye' in tag]
    idx_mouth = [i for i, tag in enumerate(tags) if 'mouth' in tag]
    idx_hair = [i for i, tag in enumerate(tags) if 'hair' in tag]
    out1 = np.where(np.array(tags) == 'head out of frame')[0][0]
    out2 = np.where(np.array(tags) == 'out of frame')[0][0]
    nohuman = np.where(np.array(tags) == 'no humans')[0][0]

    def mami_detector(prediction, path) -> bool:
        """Returns True if the image was purged"""
        n_eyes = np.max(prediction[idx_eyes])
        n_hair = np.max(prediction[idx_hair])
        n_mouth = np.max(prediction[idx_mouth])
        n_out1 = prediction[out1]
        n_out2 = prediction[out2]
        n_nohuman = prediction[nohuman]

        history = []
        flag = (n_eyes < 0.1) and (n_mouth < 0.1)
        history.append(bool(flag))
        flag |= n_hair < 0.2
        history.append(bool(flag))
        flag |= (n_out1 > 0.0025) or (n_out2 > 0.1)
        history.append(bool(flag))

        flag &= (n_eyes < 0.2) and (n_mouth < 0.2)
        history.append(bool(flag))
        flag |= (n_out1 > 0.08) or (n_out2 > 0.3)
        history.append(bool(flag))
        flag &= n_nohuman < 0.05
        if flag:
            os.makedirs(path.parent / 'mami', exist_ok=True)
            os.rename(path, path.parent / 'mami' / path.name)
            dic = {
                'eyes': n_eyes,
                'hair': n_hair,
                'mouth': n_mouth,
                'out1': n_out1,
                'out2': n_out2,
                'nohuman': n_nohuman
            }
            for k, v in dic.items():
                dic[k] = round(float(v), 4)
            dic['history'] = history
            with open(path.parent / 'mami' / (path.name[:-3] + 'json'), 'w') as f:
                json.dump(dic, f, indent=4)
            return True
        return False
    return mami_detector


if __name__ == "__main__":
    args = add_args()
    tags_path = Path(args.DD_path) / 'tags.txt'
    model_path = Path(args.DD_path) / 'model-resnet_custom_v3.h5'
    with open(tags_path) as f:
        tags = f.read().splitlines()
    replace_underscore = re.compile(r'(?<=[a-zA-Z\()])_(?=[a-zA-Z\(])')
    escape_special_char = re.compile(r'([\\():])')
    if not args.no_replace_underscore:
        tags = [replace_underscore.sub(' ', tag) for tag in tags]
    if not args.no_escape_special_char:
        tags = [escape_special_char.sub(r'\\\1', tag) for tag in tags]
    tags = np.array([tag.strip() for tag in tags])
    if args.blacklist_tags:
        for k in args.blacklist_tags:
            assert k in tags, f'{k} not in tags. The following similar tags are available: {[t for t in tags if k in t]}'
    purges = ('System', 'Character',)
    if args.tag_characters:
        purges = ('System',)
    tag_mask = get_tag_mask(args.DD_path, purges=purges)
    print(f'Tags loaded: {len(tags)} tags.')
    print(f'Purging tags of categories: {purges}')
    img_paths = list(dirwalk(Path(args.input_folder)))
    img_paths = [p for p in img_paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']]
    print(f'Found {len(img_paths)} images. Processing in {np.ceil(len(img_paths) / args.batch_size):.0f} batches.')
    nproc = args.nproc if args.nproc > 0 else None
    img_gen = dd_mt_gen(img_paths, nproc=nproc, maximum_look_ahead=args.maximum_look_ahead)
    index, scores = run_dd_keras_prediction(img_gen, model_path, batch_size=args.batch_size)
    scores = fix_order(scores, index)

    r_tags = []
    blacklist_tags = []
    blacklist_threshold = []
    n_purged = 0
    if args.blacklist_tags:
        blacklist_tags = args.blacklist_tags
        blacklist_threshold = args.blacklist_threshold
    if args.purge_text:
        text_tags = [t for t in tags if 'text' in t]
        blacklist_tags += text_tags
        blacklist_threshold += [0.1] * len(text_tags)
    purger = purger_gen(tags, blacklist_tags, blacklist_threshold)
    mami_detector = mani_detector_gen(tags)

    for i in range(len(scores)):
        if purger(scores[i], img_paths[i]):
            n_purged += 1
            continue
        elif args.purge_tomoe_mami:
            if mami_detector(scores[i], img_paths[i]):
                n_purged += 1
                continue
        mask = tag_mask & (scores[i] > args.threshold)
        idx = np.argsort(scores[i][mask])[-args.max_number_tags:]

        r_tags.append(tags[mask][idx])
        score = scores[i][idx]

        if not args.no_alpha_sort:
            r_tags[-1].sort()

        tagstr = ', '.join(r_tags[-1])
        if os.path.exists(img_paths[i].with_suffix('.txt')):
            if args.append_mode == 'skip':
                continue
            else:
                with open(img_paths[i].with_suffix('.txt'), 'r+') as f:
                    if args.append_mode == 'overwrite':
                        pass
                    elif args.append_mode == 'prepend':
                        tagstr = tagstr + f.read()
                    elif args.append_mode == 'append':
                        tagstr = f.read() + tagstr
                    f.seek(0)
                    f.truncate()
                    f.write(tagstr)
        else:
            with open(img_paths[i].with_suffix('.txt'), 'w') as f:
                f.write(tagstr)
    if n_purged > 0:
        print(f'{n_purged} images purged.')
    print(f'Written {len(r_tags)} tags to {args.input_folder}.')
