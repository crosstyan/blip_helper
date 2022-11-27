import argparse

# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse


def get_parser():
    parser = argparse.ArgumentParser()
    # general
    # https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
    parser.add_argument("-p", "--path", type=str, action="append", required=True, help="Path to the folder containing images. using `-p folder1 -p folder2` will search for images in both folder1 and folder2")
    parser.add_argument("--append", type=str, default="",
                        help="append a string to the end of the prompt. only effective when post_process is True")
    # deepdanbooru
    parser.add_argument("--no_deepdanbooru",
                        dest="deepdanbooru", action="store_false")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--alpha_sort", dest="alpha_sort", action="store_true")
    parser.add_argument("--no-use_spaces",
                        dest="use_spaces", action="store_false")
    parser.add_argument("--no-use_escape",
                        dest="use_escape", action="store_false")
    parser.add_argument("--include_ranks",
                        dest="include_ranks", action="store_true")
    parser.add_argument("--log_deepbooru", dest="log_deepbooru", action="store_true", help="show the possibility of each tag")
    # blip
    parser.add_argument("--no_blip", dest="blip", action="store_false")
    parser.add_argument("--blip_no_try_cuda", dest="running_on_cpu",
                        action="store_true")
    # parser.add_argument("--blip_use_half", dest="no_half",
    #                     action="store_false")
    # not sure if this argument is necessary
    parser.add_argument("--blip_keep_models_in_memory",
                        dest="keep_models_in_memory", action="store_true")
    parser.add_argument("--blip_num_beams", type=int, default=1)
    parser.add_argument("--blip_min_length", type=int, default=24)
    parser.add_argument("--blip_max_length", type=int, default=48)
    parser.add_argument("--blip_image_eval_size", type=int, default=384)
    parser.add_argument("--blip-use_torch_cache", dest="no_torch_cache", action="store_true",
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
