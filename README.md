# BLIP helper

A script for tagging picture with DeepDanbooru and BLIP

[Get started with tensorflow-metal](https://developer.apple.com/metal/tensorflow-plugin/) for MacOS user

## Usage

```bash
git submodule update --init --recursive
pip install -r requirements.txt
pip install -r repo/BLIP/requirements.txt
pip install -r repo/DeepDanbooru/requirements.txt
# using conda is recommended
# You would need pycocotoools
# https://anaconda.org/conda-forge/pycocotools
```

Using CPU in BLIP part by default, you can change it to GPU by `--try-cuda` option. However Tensorflow that powers DeepDanbooru would choose GPU automatically if it is available. So you might want to use GPU for DeepDanbooru and CPU for BLIP if you don't have enough VRAM.

See [`args_parser.py`](args_parser.py) for more options.

## TODO

- [ ] Add support for half precision
