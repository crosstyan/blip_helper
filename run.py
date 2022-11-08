import os.path
import re
import tempfile
import argparse
import glob
import zipfile
import numpy as np
import interrogate
import deepbooru
import args_parser

# hate it
# TODO: use another downloader
from torch.hub import download_url_to_file, urlparse, HASH_REGEX

from PIL import Image
from tqdm import tqdm
import interrogate


# Do some post processing with generated txt
# like add artist name
def post_process_prompt(prompt: str, append: str, is_prepend=True) -> str:
    if is_prepend:
        return append + ", " + prompt
    else:
        return prompt + ", " + append

def download_cached_file(url, check_hash=True, progress=False, output_dir=""):
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    models_file = os.path.join(interrogate.blip_models_folder_path, filename)
    if not os.path.exists(models_file):
        print('Downloading: "{}" to {}\n'.format(url, models_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, models_file, hash_prefix, progress=progress)
    return models_file

if __name__ == "__main__":
    # https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a
    # $env:KMP_DUPLICATE_LIB_OK=$true
    parser = args_parser.get_parser()
    args = parser.parse_args()
    if (not args.deepdanbooru) and (not args.blip):
        print("Why are you running this script?")
        exit(1)

    # check if the BLIP model exists if not download it
    if args.blip:
        if not args.use_torch_cache:
            is_blip_models_exist = os.path.exists(interrogate.blip_models_path)
            print("loading BLIP model from {}".format(interrogate.blip_models_path))
            if not is_blip_models_exist:
                download_cached_file(interrogate.blip_model_url)
        else:
            print("Using torch cache")

    # check if the deepbooru model exists if not download it
    global model
    global tags
    model = None
    tags = None
    if args.deepdanbooru:
        print("loading deepbooru model from {}".format(deepbooru.default_deepbooru_model_path))
        model, tags = deepbooru.get_deepbooru_tags_model(deepbooru.default_deepbooru_model_path)

    types = ('jpg', 'png', 'jpeg', 'gif', 'webp', 'bmp') # the tuple of file types
    p = args.path
    is_abs = os.path.isabs(args.path)
    if not is_abs:
        p = os.path.abspath(args.path)
    if not os.path.exists(p):
        print("{} not exists".format(p))
        exit(1)
    print("The picture path is {}. I will grab all the picture recursively. ".format(p))
    # copilot did this
    files_grabbed = glob.glob(os.path.join(p, "**"), recursive=True)
    print("found {} files".format(len(files_grabbed)))
    files_with_ext = [ f for f in files_grabbed if f.endswith(types) ]
    print("found {} files with extensions".format(len(files_with_ext)))
        
    interrogator = interrogate.InterrogateModels(**vars(args))
    interrogator.load()
    for image_path in tqdm(files_with_ext, desc="Processing"):
        # this should not happen. check for it anyway
        if os.path.isdir(image_path):
            continue
        image = Image.open(image_path).convert("RGB")
        prompt = ""
        # I choose to use blip first.
        if args.blip:
            prompt += interrogator.generate_caption(image)
        if args.blip and args.deepdanbooru:
            prompt += ", "
        if args.deepdanbooru:
            prompt += deepbooru.get_deepbooru_tags_from_model(
                model,
                tags,
                image,
                args.threshold,
                alpha_sort=args.alpha_sort,
                use_spaces=args.use_spaces,
                use_escape=args.use_escape,
                include_ranks=args.include_ranks,
            )
        if (args.post_process):
            prompt = post_process_prompt(prompt, args.append)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        txt_filename = os.path.join(args.path, f"{image_name}.txt")
        print(f"writing {txt_filename}: {prompt}")
        # https://stackoverflow.com/questions/4914277/how-to-empty-a-file-using-python
        # overwrite the file default
        with open(txt_filename, 'w') as f:
            f.write(prompt)
