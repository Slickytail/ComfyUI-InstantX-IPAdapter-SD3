# ComfyUI-IPAdapter-SD3

ComfyUI implementation of the [InstantX IP-Adapter for SD3.5 Large](https://huggingface.co/InstantX/SD3.5-Large-IP-Adapter).

## Installation

Download [`ip-adapter.bin` from the original repository](https://huggingface.co/InstantX/SD3.5-Large-IP-Adapter/blob/main/ip-adapter.bin), and place it in the `models/ipadapter` folder of your ComfyUI installation. (I suggest renaming it to something easier to remember).

## Usage
The IP-Adapter can be used with **Stable Diffusion 3.5 Large** and **Stable Diffusion 3.5 Large Turbo**.  
Please note that the model was originally trained on SD3.5 Large, and so the accuracy of the adapter is not as good when using the Turbo model.  
An example workflow can be found in the `workflows` directory.

I recommend using an image weight of 0.5. The start/end timestep are currently not implemented.

## TODOs
- Support for start/end timestep of IP-Adapter (easy, PRs welcome).
- Allow multiple adapters to be added together and not overwrite each other.
- Replace hardcoded parameters (such as hidden size/num layers) with values determined from the model. Would allow the same code to be used for future adapters, e.g. for SD3.5 Medium.
- Convert the adapter to safetensors.
