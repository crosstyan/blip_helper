import argparse

# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse


def get_parser():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--path", type=str, default=".")
    parser.add_argument("--post_process", type=bool, default=True)
    parser.add_argument("--append", type=str, default="sks",
                        help="append a string to the end of the prompt. only effective when post_process is True")
    # deepdanbooru
    parser.add_argument("--no-deepdanbooru",
                        dest="deepdanbooru", action="store_false")
    parser.add_argument("--threshold", type=int, default=0.75)
    parser.add_argument("--alpha_sort", dest="alpha_sort", action="store_true")
    parser.add_argument("--no-use_spaces",
                        dest="use_spaces", action="store_false")
    parser.add_argument("--no-use_escape",
                        dest="use_escape", action="store_false")
    parser.add_argument("--include_ranks",
                        dest="include_ranks", action="store_true")
    # blip
    parser.add_argument("--no-blip", dest="blip", action="store_false")
    parser.add_argument("--try-cuda", dest="running_on_cpu",
                        action="store_false")
    parser.add_argument("--use_half", dest="no_half", action="store_false")
    parser.add_argument("--keep_models_in_memory",
                        dest="keep_models_in_memory", action="store_true")
    parser.add_argument("--blip_num_beams", type=int, default=1)
    parser.add_argument("--blip_min_length", type=int, default=24)
    parser.add_argument("--blip_max_length", type=int, default=48)
    parser.add_argument("--blip_image_eval_size", type=int, default=384)
    parser.add_argument("--use_torch_cache", type=bool, default=False,
                        help="use torch cache directory to download the model instead of default pretrained/BLIP")
    parser.set_defaults(
        alpha_sort=False,
        use_spaces=True,
        use_escape=True,
        include_ranks=False,
        running_on_cpu=True,
        no_half=True,
        keep_models_in_memory=False,
        blip=True,
        deepdanbooru=True)
    return parser
