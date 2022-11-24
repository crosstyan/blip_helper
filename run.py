import os.path
from psutil import cpu_count
from pathlib import Path
import re
import tempfile
import argparse
import glob
import zipfile
import numpy as np
import blip
import deepbooru_utils
import fast_deepdanbooru
import args_parser

from PIL import Image
from tqdm import tqdm
import blip

class ImageAnnotation:
    def __init__(self, image_path: str):
        self.image_path:Path = Path(image_path)
        # will be filled by DeepDanbooru
        self.tags:list[str] = []
        # will be filled by BLIP
        self.prompt = ""

    def __str__(self):
        return "image {}".format(self.image_path)

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
        print("loading deepbooru model from {}".format(deepbooru_utils.default_deepbooru_model_path))
        model, tags = deepbooru_utils.init_deepbooru()

    # gif is leave out intentional 
    types = ['jpg', 'png', 'jpeg', 'webp', 'bmp']
    p = args.path
    is_abs = os.path.isabs(args.path)
    if not is_abs:
        p = os.path.abspath(args.path)
    if not os.path.exists(p):
        print("{} not exists".format(p))
        exit(1)
    print("The picture path is \"{}\". I will grab all the picture recursively. ".format(p))
    # copilot did this
    files_grabbed = glob.glob(os.path.join(p, "**"), recursive=True)
    files_grabbed = map(Path, files_grabbed)
    print("found {} files".format(len(files_grabbed)))
    files_with_ext = [ f for f in files_grabbed if f.suffix.lower() in types ]
    print("found {} files with extensions".format(len(files_with_ext)))
    imgs = map(ImageAnnotation, files_with_ext)
    # Number of processes in processing module. Defaults to number of CPU cores
    nproc = cpu_count()
    # Maximum number of images to preload
    maximum_look_ahead = 128
    batch_size = 4
    img_gen = fast_deepdanbooru.dd_mt_gen(files_grabbed, nproc=nproc, maximum_look_ahead=maximum_look_ahead)
    index, scores = fast_deepdanbooru.run_dd_keras_prediction(img_gen, model, batch_size=batch_size)
    scores = fast_deepdanbooru.fix_order(scores.index)
        
    # for image_path in tqdm(imgs, desc="Processing"):
    #     # this should not happen. check for it anyway
    #     if os.path.isdir(image_path):
    #         continue
    #     # image = Image.open(image_path).convert("RGB")
    #     # prompt = ""
    #     # # I choose to use blip first.
    #     # if args.blip:
    #     #     prompt += interrogator.generate_caption(image)
    #     # if args.blip and args.deepdanbooru:
    #     #     prompt += ", "
    #     # if args.deepdanbooru:
    #     #     prompt += deepbooru.get_deepbooru_tags_from_model(
    #     #         model,
    #     #         tags,
    #     #         image,
    #     #         args.threshold,
    #     #         alpha_sort=args.alpha_sort,
    #     #         use_spaces=args.use_spaces,
    #     #         use_escape=args.use_escape,
    #     #         include_ranks=args.include_ranks,
    #     #         log_results=args.log_deepbooru,
    #     #     )
    #     if (args.append != ""):
    #         prompt = post_process_prompt(prompt, args.append)
    #     image_name = os.path.splitext(os.path.basename(image_path))[0]
    #     txt_filename = os.path.join(args.path, f"{image_name}.txt")
    #     print(f"\nwriting {txt_filename}: {prompt}\n")
    #     # https://stackoverflow.com/questions/4914277/how-to-empty-a-file-using-python
    #     # overwrite the file default
    #     with open(txt_filename, 'w') as f:
    #         f.write(prompt)
