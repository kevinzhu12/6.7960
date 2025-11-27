import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image

# Initialize CUDA and cuDNN
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("cuDNN enabled:", torch.backends.cudnn.enabled)
print("cuDNN version:", torch.backends.cudnn.version())
print("CUDA version:", torch.version.cuda)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# Load pipeline
device = "cuda"
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to(device)

# input_image_path = "outputs/output_text2img.png"
input_image_path = "datasets/COD10K-v3/Train/Image/COD10K-CAM-1-Aquatic-1-BatFish-1.jpg"
init_image = load_image(input_image_path).convert("RGB")

init_image = init_image.resize((512, 512))
# add white background
# white_bg = Image.new("RGBA", init_image.size, "WHITE")
# white_bg.paste(init_image, (0,0), init_image)
# result_img = white_bg.convert("RGB")

# result_img.save("intermediate.png")

# prompt = input("Modification from text2img.png: ")
# prompt = 'enhance the animal in the image to enable a classifier to easily identify it'
prompt = ''
# strength determines how much of original image is modified (0=None, 1=New)
strength = 0.2

image = pipe(
    prompt=prompt,
    image=init_image,
    strength=strength,
    guidance_scale=3 # how strongly to follow text prompt
).images[0]

image.save("outputs/no_prompt_batfish_1_0.2_strength_3_guidance_img2img.png")
