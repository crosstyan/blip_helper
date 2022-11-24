import argparse

# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse


def get_parser():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--path", type=str, default=".")
    parser.add_argument("--append", type=str, default="",
                        help="append a string to the end of the prompt. only effective when post_process is True")
    # deepdanbooru
    parser.add_argument("--no-deepdanbooru",
                        dest="deepdanbooru", action="store_false")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--alpha-sort", dest="alpha_sort", action="store_true")
    parser.add_argument("--no-use-spaces",
                        dest="use_spaces", action="store_false")
    parser.add_argument("--no-use-escape",
                        dest="use_escape", action="store_false")
    parser.add_argument("--include-ranks",
                        dest="include-ranks", action="store_true")
    parser.add_argument("--log-deepbooru", dest="log_deepbooru", action="store_true", help="show the possibility of each tag")
    # blip
    parser.add_argument("--no_blip", dest="blip", action="store_false")
    parser.add_argument("--blip-no-try_cuda", dest="running_on_cpu",
                        action="store_true")
    # parser.add_argument("--blip_use_half", dest="no_half",
    #                     action="store_false")
    # not sure if this argument is necessary
    parser.add_argument("--blip-keep_models-in_memory",
                        dest="keep_models_in_memory", action="store_true")
    parser.add_argument("--blip-num-beams", type=int, default=1)
    parser.add_argument("--blip-min-length", type=int, default=24)
    parser.add_argument("--blip-max-length", type=int, default=48)
    parser.add_argument("--blip-image-eval-size", type=int, default=384)
    parser.add_argument("--blip-use-torch-cache", dest="no_torch_cache", action="store_true",
                        help="use torch cache directory to download the model instead of default pretrained/BLIP")
    parser.set_defaults(
        log_deepdanbooru=False,
        alpha_sort=False,
        use_spaces=True,
        use_escape=True,
        include_ranks=False,
        running_on_cpu=False,
        no_half=True,
        keep_models_in_memory=False,
        use_torch_cache=False,
        blip=True,
        deepdanbooru=True)
    return parser
