import contextlib
import os
import sys
import traceback
from collections import namedtuple
import re

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
deivces_interrogate = torch.device("cuda")


pwd = os.path.dirname(os.path.realpath(__file__))
clip_models_path = os.path.join(pwd, "models")

# "interrogate_return_ranks": OptionInfo(False, "Interrogate: include ranks of model tags matches in results (Has no effect on caption-based interrogators)."),
# "interrogate_clip_num_beams": OptionInfo(1, "Interrogate: num_beams for BLIP", gr.Slider, {"minimum": 1, "maximum": 16, "step": 1}),
# "interrogate_clip_min_length": OptionInfo(24, "Interrogate: minimum description length (excluding artists, etc..)", gr.Slider, {"minimum": 1, "maximum": 128, "step": 1}),
# "interrogate_clip_max_length": OptionInfo(48, "Interrogate: maximum description length", gr.Slider, {"minimum": 1, "maximum": 256, "step": 1}),
# "interrogate_clip_dict_limit": OptionInfo(1500, "CLIP: maximum number of lines in text file (0 = No limit)"),

is_running_on_cpu = False
is_no_half = False
interrogate_keep_models_in_memory = False
interrogate_return_ranks = False
interrogate_clip_dict_limit:int = 0
interrogate_clip_num_beams:int = 1
interrogate_clip_min_length:int = 24
interrogate_clip_max_length:int = 48
interrogate_clip_dict_limit:int = 1500
med_config = os.path.join(pwd, "repo", "BLIP", "configs", "med_config.json")


blip_image_eval_size = 384
blip_model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth'
clip_model_name = 'ViT-L/14'

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
        import models.blip

        blip_model = models.blip.blip_decoder(pretrained=blip_model_url, 
                                              image_size=blip_image_eval_size, vit='base', 
                                              med_config=med_config)
        blip_model.eval()

        return blip_model

    def load_clip_model(self):
        import clip

        if self.running_on_cpu:
            model, preprocess = clip.load(clip_model_name, device="cpu", download_root=clip_models_path)
        else:
            model, preprocess = clip.load(clip_model_name, download_root=clip_models_path)

        model.eval()
        model = model.to(deivces_interrogate)

        return model, preprocess

    def load(self):
        if self.blip_model is None:
            self.blip_model = self.load_blip_model()
            if not is_no_half and not self.running_on_cpu:
                self.blip_model = self.blip_model.half()

        self.blip_model = self.blip_model.to(deivces_interrogate)

        if self.clip_model is None:
            self.clip_model, self.clip_preprocess = self.load_clip_model()
            if not is_no_half and not self.running_on_cpu:
                self.clip_model = self.clip_model.half()

        self.clip_model = self.clip_model.to(deivces_interrogate)

        self.dtype = next(self.clip_model.parameters()).dtype

    def send_clip_to_ram(self):
        if not interrogate_keep_models_in_memory:
            if self.clip_model is not None:
                self.clip_model = self.clip_model.to(cpu)

    def send_blip_to_ram(self):
        if not interrogate_keep_models_in_memory:
            if self.blip_model is not None:
                self.blip_model = self.blip_model.to(cpu)

    def unload(self):
        self.send_clip_to_ram()
        self.send_blip_to_ram()

        torch_gc()

    def rank(self, image_features, text_array, top_count=1):
        import clip

        if interrogate_clip_dict_limit != 0:
            text_array = text_array[0:int(interrogate_clip_dict_limit)]

        top_count = min(top_count, len(text_array))
        text_tokens = clip.tokenize([text for text in text_array], truncate=True).to(deivces_interrogate)
        text_features = self.clip_model.encode_text(text_tokens).type(self.dtype)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = torch.zeros((1, len(text_array))).to(deivces_interrogate)
        for i in range(image_features.shape[0]):
            similarity += (100.0 * image_features[i].unsqueeze(0) @ text_features.T).softmax(dim=-1)
        similarity /= image_features.shape[0]

        top_probs, top_labels = similarity.cpu().topk(top_count, dim=-1)
        return [(text_array[top_labels[0][i].numpy()], (top_probs[0][i].numpy()*100)) for i in range(top_count)]

    def generate_caption(self, pil_image):
        gpu_image = transforms.Compose([
            transforms.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(pil_image).unsqueeze(0).type(self.dtype).to(deivces_interrogate)

        with torch.no_grad():
            caption = self.blip_model.generate(gpu_image, sample=False, num_beams=interrogate_clip_num_beams, min_length=interrogate_clip_min_length, max_length=interrogate_clip_max_length)

        return caption[0]

    def interrogate(self, pil_image):
        res = None

        try:

            # if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
            #     lowvram.send_everything_to_cpu()
            #     devices.torch_gc()

            self.load()

            caption = self.generate_caption(pil_image)
            self.send_blip_to_ram()
            torch_gc()

            res = caption

            clip_image = self.clip_preprocess(pil_image).unsqueeze(0).type(self.dtype).to(deivces_interrogate)

            # precision_scope = torch.autocast if shared.cmd_opts.precision == "autocast" else contextlib.nullcontext
            # Don't care about precision for now
            precision_scope = torch.autocast 
            with torch.no_grad(), precision_scope("cuda"):
                image_features = self.clip_model.encode_image(clip_image).type(self.dtype)

                image_features /= image_features.norm(dim=-1, keepdim=True)

                # if shared.opts.interrogate_use_builtin_artists:
                #     artist = self.rank(image_features, ["by " + artist.name for artist in shared.artist_db.artists])[0]

                #     res += ", " + artist[0]

                for name, topn, items in self.categories:
                    matches = self.rank(image_features, items, top_count=topn)
                    for match, score in matches:
                        if interrogate_return_ranks:
                            res += f", ({match}:{score/100:.3f})"
                        else:
                            res += ", " + match

        except Exception:
            print(f"Error interrogating", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            res += "<error>"

        self.unload()

        return res
