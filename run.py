import os.path
import re
import tempfile
import argparse
import glob
import zipfile
import numpy as np
import blip
import deepbooru
import args_parser
from pathlib import Path

from PIL import Image
from tqdm import tqdm
import blip

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Do some post processing with generated txt
# like add artist name
def post_process_prompt(prompt: str, append: str, is_prepend=True) -> str:
    if is_prepend:
        return append + ", " + prompt
    else:
        return prompt + ", " + append

if __name__ == "__main__":
    # https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a
    # $env:KMP_DUPLICATE_LIB_OK=$true
    parser = args_parser.get_parser()
    args = parser.parse_args()
    if (not args.deepdanbooru) and (not args.blip):
        print("Why are you running this script?")
        exit(1)

    types = ['jpg', 'png', 'jpeg', 'gif', 'webp', 'bmp'] # the tuple of file types
    files_with_ext:list[Path] = []
    for p in args.path:
        p = Path(p)
        if not p.exists():
            print("{} not exists".format(p))
            continue
        print("The picture path is \"{}\". I will grab all the picture recursively. ".format(p))
        files_grabbed = p.glob('**/*')
        # suffix is the file extension with the dot
        _files_with_ext = [ f for f in files_grabbed if f.suffix[1:] in types ]
        print("found {} files with extensions".format(len(_files_with_ext)))
        files_with_ext.extend(_files_with_ext)
    print("Find totally {} files with extensions".format(len(files_with_ext)))

    # check if the BLIP model exists if not download it and then initialize blip
    global interrogator
    interrogator = None
    if args.blip:
        if not args.use_torch_cache:
            is_blip_models_exist = os.path.exists(blip.blip_models_path)
            print("loading BLIP model from {}".format(blip.blip_models_path))
            if not is_blip_models_exist:
                blip.download_blip_models()
        else:
            print("Using torch cache")
        interrogator = blip.InterrogateModels(**vars(args))
        interrogator.load()

    # check if the deepbooru model exists if not download it
    global model
    global tags
    model = None
    tags = None
    if args.deepdanbooru:
        print("loading deepbooru model from {}".format(deepbooru.default_deepbooru_model_path))
        model, tags = deepbooru.init_deepbooru()
        
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
                log_results=args.log_deepbooru,
            )
        if (args.append != ""):
            prompt = post_process_prompt(prompt, args.append)
        txt_filename = image_path.with_suffix(".txt")
        print(f"\nwriting {txt_filename}: {prompt}\n")
        # https://stackoverflow.com/questions/4914277/how-to-empty-a-file-using-python
        # overwrite the file default
        with open(txt_filename, 'w') as f:
            f.write(prompt)
