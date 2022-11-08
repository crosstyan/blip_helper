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
python run.py --path /path/to/image
```

Using CPU in BLIP part by default, you can change it to GPU by `--blip-try-cuda` option. However Tensorflow that powers DeepDanbooru would choose GPU automatically if it is available. So you might want to use GPU for DeepDanbooru and CPU for BLIP if you don't have enough VRAM.

It's blazingly fast when use DeepDanbooru OR BLIP when using GPU. But it's slow when use both of them. 
I guess these's a bug in DeepDanbooru since it eats up all my VRAM. I'll try to fix it. (Maybe it's a feature) (Maybe I could load it to memory to let it eats up less VRAM?) Needs more investigation.


See [`args_parser.py`](args_parser.py) for more options.

## TODO

- [ ] Add support for half precision for BLIP
- [ ] Add support for choosing GPU for BLIP (trivial but why?)
- [ ] A workflow for tagging
- [ ] filter out unwanted tags like orginal anime character
