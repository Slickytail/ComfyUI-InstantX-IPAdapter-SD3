import os
import logging
from functools import partial

import torch
import folder_paths
from transformers import AutoProcessor, SiglipVisionModel

from .models.resampler import TimeResampler
from .models.jointblock import JointBlockIPWrapper, IPAttnProcessor

MODELS_DIR = os.path.join(folder_paths.models_dir, "ipadapter")
if "ipadapter" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["ipadapter"]
folder_paths.folder_names_and_paths["ipadapter"] = (
    current_paths,
    folder_paths.supported_pt_extensions,
)


def patch(patcher, ip_procs, resampler: TimeResampler, clip_embeds, weight=1.0):
    """
    Patches a model_sampler to add the ipadapter
    """
    mmdit = patcher.model.diffusion_model
    timestep_schedule_max = patcher.model.model_config.sampling_settings.get(
        "timesteps", 1000
    )
    # hook the model's forward function
    # so that when it gets called, we can grab the timestep and send it to the resampler
    ip_options = {
        "hidden_states": None,
        "t_emb": None,
        "t": None,
        "weight": weight,
        "cond_or_uncond": None,
    }

    def ddit_wrapper(forward, args):
        batch_size = args["input"].shape[0] // len(args["cond_or_uncond"])
        # if we're only doing cond or only doing uncond, only pass one of them through the resampler
        embeds = clip_embeds[args["cond_or_uncond"]]
        # then, we can repeat the embeds to the batch size
        embeds = torch.repeat_interleave(embeds, batch_size, dim=0)
        # the resampler wants between 0 and MAX_STEPS
        timestep = args["timestep"] * timestep_schedule_max
        image_emb, t_emb = resampler(embeds, timestep, need_temb=True)
        # these will need to be accessible to the IPAdapters
        ip_options["hidden_states"] = image_emb
        ip_options["t_emb"] = t_emb
        # this is between 0 and 1, so the adapters can calculate start_point and end_point
        # actually, do we need to get the sigma value instead?
        ip_options["t"] = args["timestep"]
        ip_options["cond_or_uncond"] = args["cond_or_uncond"]

        return forward(args["input"], args["timestep"], **args["c"])

    patcher.set_model_unet_function_wrapper(ddit_wrapper)
    # patch each dit block
    for i, block in enumerate(mmdit.joint_blocks):
        wrapper = JointBlockIPWrapper(block, ip_procs[i], ip_options)
        patcher.set_model_patch_replace(wrapper, "dit", "double_block", i)


class SD3IPAdapter:
    def __init__(self, encoder_path: str, checkpoint: str, device):
        self.device = device
        # load image encoder
        self.encoder = SiglipVisionModel.from_pretrained(encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.processor = AutoProcessor.from_pretrained(encoder_path)
        # load the checkpoint right away
        self.state_dict = torch.load(
            os.path.join(MODELS_DIR, checkpoint),
            map_location=self.device,
            weights_only=True,
        )
        # todo: infer some of the params from the checkpoint instead of hardcoded
        self.resampler = TimeResampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=64,
            embedding_dim=1152,
            output_dim=2432,
            ff_mult=4,
            timestep_in_dim=320,
            timestep_flip_sin_to_cos=True,
            timestep_freq_shift=0,
        )
        self.resampler.eval()
        self.resampler.to(self.device, dtype=torch.float16)
        self.resampler.load_state_dict(self.state_dict["image_proj"])

        # now we'll create the attention processors
        # ip_adapter.keys looks like [0.proj, 0.to_k, ..., 1.proj, 1.to_k, ...]
        n_procs = len(
            set(x.split(".")[0] for x in self.state_dict["ip_adapter"].keys())
        )
        self.procs = torch.nn.ModuleList(
            [
                # this is hardcoded for SD3.5L
                IPAttnProcessor(
                    hidden_size=2432,
                    cross_attention_dim=2432,
                    ip_hidden_states_dim=2432,
                    ip_encoder_hidden_states_dim=2432,
                    head_dim=64,
                    timesteps_emb_dim=1280,
                ).to(self.device, dtype=torch.float16)
                for _ in range(n_procs)
            ]
        )
        self.procs.load_state_dict(self.state_dict["ip_adapter"])

    @torch.inference_mode()
    def encode(self, image):
        clip_image = self.processor.image_processor(
            image, return_tensors="pt", do_rescale=False
        ).pixel_values
        clip_image_embeds = self.encoder(
            clip_image.to(self.device, dtype=self.encoder.dtype),
            output_hidden_states=True,
        ).hidden_states[-2]
        clip_image_embeds = torch.cat(
            [clip_image_embeds, torch.zeros_like(clip_image_embeds)], dim=0
        )
        clip_image_embeds = clip_image_embeds.to(dtype=torch.float16)
        return clip_image_embeds


class IPAdapterSD3Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": (folder_paths.get_filename_list("ipadapter"),),
                "clip_vision": (["google/siglip-so400m-patch14-384"],),
                "provider": (["cuda", "cpu", "mps"],),
            }
        }

    RETURN_TYPES = ("IP_ADAPTER_SD3_INSTANTX",)
    RETURN_NAMES = ("ipadapter",)
    FUNCTION = "load_model"
    CATEGORY = "InstantXNodes"

    def load_model(self, ipadapter, clip_vision, provider):
        logging.info("Loading InstantX IPAdapter SD3 model.")
        model = SD3IPAdapter(clip_vision, ipadapter, provider)
        return (model,)


class ApplyIPAdapterSD3:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter": ("IP_ADAPTER_SD3_INSTANTX",),
                "image": ("IMAGE",),
                "weight": (
                    "FLOAT",
                    {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"
    CATEGORY = "InstantXNodes"

    def apply_ipadapter(
        self, model, ipadapter, image, weight, start_percent, end_percent
    ):
        # set model
        new_model = model.clone()
        embeds = ipadapter.encode(image)
        patch(new_model, ipadapter.procs, ipadapter.resampler, embeds, weight=weight)
        return (new_model,)


NODE_CLASS_MAPPINGS = {
    "IPAdapterSD3Loader": IPAdapterSD3Loader,
    "ApplyIPAdapterSD3": ApplyIPAdapterSD3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapterSD3Loader": "Load IPAdapter SD3 Model",
    "ApplyIPAdapterSD3": "Apply IPAdapter SD3 Model",
}
