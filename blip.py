import os.path
import re
import tempfile
import argparse
import glob
import zipfile
import deepdanbooru as dd
import tensorflow as tf
import numpy as np
import interrogate

from PIL import Image
from tqdm import tqdm

# Do some post processing with generated txt
# like add artist name
def post_process_prompt(prompt: str, append: str) -> str:
    prompt = prompt + ", " + append
    return prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=".")
    parser.add_argument("--threshold", type=int, default=0.75)
    parser.add_argument("--alpha_sort", type=bool, default=False)
    parser.add_argument("--use_spaces", type=bool, default=True)
    parser.add_argument("--use_escape", type=bool, default=True)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--include_ranks", type=bool, default=False)
    parser.add_argument("--post_process", type=bool, default=True)
    parser.add_argument("--append", type=str, default="sks", help="append a string to the end of the prompt. only effective when post_process is True")

    args = parser.parse_args()

    global model_path
    model_path:str
    if args.model_path == "":
        script_path = os.path.realpath(__file__)
        default_model_path = os.path.join(os.path.dirname(script_path), "models")
        print("No model path specified, using default model path: {}".format(default_model_path))
        model_path = default_model_path
    else:
        model_path = args.model_path

    types = ('jpg', 'png', 'jpeg', 'gif', 'webp', 'bmp') # the tuple of file types
    p = args.path
    is_abs = os.path.isabs(args.path)
    if not is_abs:
        p = os.path.abspath(args.path)
    if not os.path.exists(p):
        print("{} not exists".format(p))
        exit(1)
    print("abs path is {}".format(p))
    # copilot did this
    files_grabbed = glob.glob(os.path.join(p, "**"), recursive=True)
    print("found {} files".format(len(files_grabbed)))
    files_with_ext = [ f for f in files_grabbed if f.endswith(types) ]
    print("found {} files with extensions".format(len(files_with_ext)))
        
    interrogator = interrogate.InterrogateModels("interrogate")
    interrogator.load()
    for image_path in tqdm(files_with_ext, desc="Processing"):
        if os.path.isdir(image_path):
            continue
        image = Image.open(image_path).convert("RGB")
        prompt = interrogator.generate_caption(image)
        if (args.post_process):
            prompt = post_process_prompt(prompt, args.append)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        txt_filename = os.path.join(args.path, f"{image_name}.txt")
        print(f"writing {txt_filename}: {prompt}")
        # https://stackoverflow.com/questions/4914277/how-to-empty-a-file-using-python
        # overwrite the file default
        with open(txt_filename, 'w') as f:
            f.write(prompt)