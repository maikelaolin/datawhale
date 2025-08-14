import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

cache_dir = "" #这里是放模型的路径
pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16, cache_dir=cache_dir)
pipe.to("cuda")

def tieditor(input_image, prompt, prompt_2):
    image = pipe(
    image=input_image,
    prompt=prompt,
    prompt_2=prompt_2,
    guidance_scale=2.5
    ).images[0]
    return image
