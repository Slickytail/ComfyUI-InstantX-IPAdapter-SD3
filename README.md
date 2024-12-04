# ComfyUI-IPAdapter-SD3

ComfyUI implementation of the [InstantX IP-Adapter for SD3.5 Large](https://huggingface.co/InstantX/SD3.5-Large-IP-Adapter).

## Installation

Download [`ip-adapter.bin` from the original repository](https://huggingface.co/InstantX/SD3.5-Large-IP-Adapter/blob/main/ip-adapter.bin), and place it in the `models/ipadapter` folder of your ComfyUI installation.

## TODOs
- Support for start/end timestep of IP-Adapter.
- Allow multiple adapters to be added together and not overwrite each other.
- Replace hardcoded parameters (such as hidden size/num layers) with values determined from the model. Would allow the same code to be used for future adapters, e.g. for SD3.5 Medium.
- Convert the adapter to safetensors.
