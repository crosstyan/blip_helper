import contextlib
import os
import sys
import traceback
from collections import namedtuple
import re
import importlib as imp
import torch

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# has_mps is only available in nightly pytorch (for now), `getattr` for compatibility
has_mps = getattr(torch, 'has_mps', False)
# if device_id is not None:
#     cuda_device = f"cuda:{device_id}"
#     return torch.device(cuda_device)
cpu = torch.device("cpu")
# deivces_interrogate = torch.device("cuda")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deivces_interrogate = cpu

pwd = os.path.dirname(os.path.realpath(__file__))
# must be the root directory of the BLIP repo
blip_dir = os.path.join(pwd, "repo", "BLIP")
sys.path.insert(0, blip_dir)

# "interrogate_return_ranks": OptionInfo(False, "Interrogate: include ranks of model tags matches in results (Has no effect on caption-based interrogators)."),
# "interrogate_clip_num_beams": OptionInfo(1, "Interrogate: num_beams for BLIP", gr.Slider, {"minimum": 1, "maximum": 16, "step": 1}),
# "interrogate_clip_min_length": OptionInfo(24, "Interrogate: minimum description length (excluding artists, etc..)", gr.Slider, {"minimum": 1, "maximum": 128, "step": 1}),
# "interrogate_clip_max_length": OptionInfo(48, "Interrogate: maximum description length", gr.Slider, {"minimum": 1, "maximum": 256, "step": 1}),
# "interrogate_clip_dict_limit": OptionInfo(1500, "CLIP: maximum number of lines in text file (0 = No limit)"),

is_running_on_cpu = True
is_no_half = False
keep_models_in_memory = False
blip_num_beams:int = 1
blip_min_length:int = 24
blip_max_length:int = 48
blip_image_eval_size = 384
# interrogate_clip_dict_limit:int = 0
# interrogate_clip_dict_limit:int = 1500
use_torch_cache = False
med_config = os.path.join(blip_dir, "configs", "med_config.json")

blip_model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth'
blip_model_filename = "model_base_caption_capfilt_large.pth"

# the folder containing the models
blip_models_folder_path = os.path.join(pwd, "pretrained", "BLIP")

# the path of pth
blip_models_path = os.path.join(blip_models_folder_path, blip_model_filename)

Category = namedtuple("Category", ["name", "topn", "items"])

re_topn = re.compile(r"\.top(\d+)\.")


class InterrogateModels:
    blip_model = None
    clip_model = None
    clip_preprocess = None
    categories = None
    dtype = None
    running_on_cpu = None

    def __init__(self, content_dir):
        self.categories = []
        # self.running_on_cpu = deivces_interrogate == torch.device("cpu")
        self.running_on_cpu = is_running_on_cpu

        if os.path.exists(content_dir):
            for filename in os.listdir(content_dir):
                m = re_topn.search(filename)
                topn = 1 if m is None else int(m.group(1))

                with open(os.path.join(content_dir, filename), "r", encoding="utf8") as file:
                    lines = [x.strip() for x in file.readlines()]

                self.categories.append(Category(name=filename, topn=topn, items=lines))

    def load_blip_model(self):
        # if you can't find this you have to install the
        # requirements.txt in the repo/BLIP folder
        # pycocotools is needed
        import models.blip

        model_path_or_url =  blip_model_url if use_torch_cache else blip_models_path 

        # blip_model = models.blip.blip_decoder(pretrained=blip_model_url, 
        #                                       image_size=blip_image_eval_size, vit='base', 
        #                                       med_config=med_config)

        # NOTE model_base_caption_capfilt_large.pth should exist in the pretrained folder
        blip_model = models.blip.blip_decoder(pretrained=model_path_or_url, 
                                              image_size=blip_image_eval_size, vit='base', 
                                              med_config=med_config)
        blip_model.eval()

        return blip_model

    def load(self):
        if self.blip_model is None:
            self.blip_model = self.load_blip_model()
            if not is_no_half and not self.running_on_cpu:
                self.blip_model = self.blip_model.half()

        self.blip_model = self.blip_model.to(deivces_interrogate)

    def send_blip_to_ram(self):
        if not keep_models_in_memory:
            if self.blip_model is not None:
                self.blip_model = self.blip_model.to(cpu)

    def unload(self):
        self.send_blip_to_ram()
        torch_gc()

    def generate_caption(self, pil_image):
        gpu_image = transforms.Compose([
            transforms.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(pil_image).unsqueeze(0).to(deivces_interrogate)

        with torch.no_grad():
            caption = self.blip_model.generate(gpu_image, sample=False, num_beams=blip_num_beams, min_length=blip_min_length, max_length=blip_max_length)
        return caption[0]
