# BLIP helper

A script for tagging picture with DeepDanbooru and BLIP

This script is NOT intended to be publish to [PyPI](https://pypi.org). Clone this repository and run it directly,
since I can't make sense of the package management of Python. Help wanted! 

## Usage

```bash
git clone https://github.com/crosstyan/blip_helper
cd blip_helper
git submodule update --init --recursive
pip install -r requirements.txt
pip install repo/DeepDanbooru
# using conda is recommended
# You would need pycocotoools
# https://anaconda.org/conda-forge/pycocotools
python run.py --path /path/to/image
```

See [`args_parser.py`](args_parser.py) for more options.

## TODO

- [ ] Add support for metal for BLIP (based on PyTorch)
- [ ] Add support for half precision for BLIP
- [ ] Add support for choosing GPU for BLIP (trivial but why?)
- [ ] A workflow for tagging
- [ ] filter out unwanted tags like orginal anime character

## Misc

- [Get started with tensorflow-metal](https://developer.apple.com/metal/tensorflow-plugin/) for MacOS user

### Choose Catagories

check `repo/DeepDanbooru/deepdanbooru/commands/download_tags.py`

```python
category_definitions = [
    {
        "category_name": "General",
        "category": "general",
        "path": os.path.join(project_path, "tags-general.txt"),
    },
    {
       'category_name': 'Artist',
       'category': 'artist',
       'path': os.path.join(path, 'tags-artist.txt'),
    },
    {
       'category_name': 'Copyright',
       'category': 'copyright',
       'path': os.path.join(path, 'tags-copyright.txt'),
    },
    {
        "category_name": "Character",
        "category": "character",
        "path": os.path.join(project_path, "tags-character.txt"),
    },
]
```

Comment out the ones you don't want to apply.
Usually I just leave `general` on.

### MacOS 

- [arm64 support for M1 ](https://github.com/tensorflow/io/issues/1298)

```bash
git clone https://github.com/tensorflow/io tfio
cd tfio
python3 setup.py -q bdist_wheel --project tensorflow_io_gcs_filesystem
pip install .
```
