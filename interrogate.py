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


pwd = os.path.dirname(os.path.realpath(__file__))
# must be the root directory of the BLIP repo
blip_dir = os.path.join(pwd, "repo", "BLIP")
sys.path.insert(0, blip_dir)

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
    running_on_cpu = None

    def __init__(self,
                 no_half=False,
                 running_on_cpu=True,
                 keep_models_in_memory=False,
                 use_torch_cache=False,
                 blip_num_beams=1,
                 blip_min_length=24,
                 blip_max_length=48,
                 blip_image_eval_size=384,
                 **kwargs
                 ):
        print("running on cpu", running_on_cpu)
        self.running_on_cpu = running_on_cpu
        self.no_half = no_half
        self.keep_models_in_memory = keep_models_in_memory
        self.blip_num_beams = blip_num_beams
        self.blip_min_length = blip_min_length
        self.blip_max_length = blip_max_length
        self.blip_image_eval_size = blip_image_eval_size
        self.use_torch_cache = use_torch_cache

        # has_mps is only available in nightly pytorch (for now), `getattr` for compatibility
        # has_mps = getattr(torch, 'has_mps', False)
        # specify the device to use by device id
        # if device_id is not None:
        #     cuda_device = f"cuda:{device_id}"
        #     return torch.device(cuda_device)

        # You must have a CPU, I guess
        self.cpu = torch.device("cpu")
        cpu = self.cpu
        if self.running_on_cpu:
            self.deivces_interrogate = cpu
        else:
            if torch.cuda.is_available():
                torch.device('cuda')
            else:
                print("CUDA is not available, using CPU")
                self.deivces_interrogate = cpu

    def load_blip_model(self):
        # if you can't find this you have to install the
        # requirements.txt in the repo/BLIP folder
        # pycocotools is needed
        import models.blip

        model_path_or_url = blip_model_url if self.use_torch_cache else blip_models_path

        # NOTE model_base_caption_capfilt_large.pth should exist in the pretrained folder
        blip_model = models.blip.blip_decoder(pretrained=model_path_or_url,
                                              image_size=self.blip_image_eval_size, vit='base',
                                              med_config=med_config)
        blip_model.eval()

        return blip_model

    def load(self):
        if self.blip_model is None:
            self.blip_model = self.load_blip_model()
            if not self.no_half and not self.running_on_cpu:
                self.blip_model = self.blip_model.half()

        self.blip_model = self.blip_model.to(self.deivces_interrogate)

    def send_blip_to_ram(self):
        if not self.keep_models_in_memory:
            if self.blip_model is not None:
                self.blip_model = self.blip_model.to(self.cpu)

    def unload(self):
        self.send_blip_to_ram()
        torch_gc()

    def generate_caption(self, pil_image):
        gpu_image = transforms.Compose([
            transforms.Resize((self.blip_image_eval_size, self.blip_image_eval_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(pil_image).unsqueeze(0).to(self.deivces_interrogate)

        with torch.no_grad():
            caption = self.blip_model.generate(
                gpu_image, sample=False, num_beams=self.blip_num_beams, min_length=self.blip_min_length, max_length=self.blip_max_length)
        return caption[0]
