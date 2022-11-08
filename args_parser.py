import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    # general 
    parser.add_argument("--path", type=str, default=".")
    parser.add_argument("--post_process", type=bool, default=True)
    parser.add_argument("--append", type=str, default="sks", help="append a string to the end of the prompt. only effective when post_process is True")
    # deepdanbooru
    parser.add_argument("--deepdanbooru", type=bool, default=True)
    parser.add_argument("--threshold", type=int, default=0.75)
    parser.add_argument("--alpha_sort", type=bool, default=False)
    parser.add_argument("--use_spaces", type=bool, default=True)
    parser.add_argument("--use_escape", type=bool, default=True)
    parser.add_argument("--include_ranks", type=bool, default=False)
    # blip
    parser.add_argument("--blip", type=bool, default=True)
    parser.add_argument("--running_on_cpu", type=bool, default=True)
    parser.add_argument("--no_half", type=bool, default=False)
    parser.add_argument("--keep_models_in_memory", type=bool, default=False)
    parser.add_argument("--blip_num_beams", type=int, default=1)
    parser.add_argument("--blip_min_length", type=int, default=24)
    parser.add_argument("--blip_max_length", type=int, default=48)
    parser.add_argument("--blip_image_eval_size", type=int, default=384)
    parser.add_argument("--use_torch_cache", type=bool, default=False, help="use torch cache directory to download the model instead of default pretrained/BLIP")
    return parser