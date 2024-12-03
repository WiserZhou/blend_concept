import torch
from models.CustomSD import CustomSDPipeline
from models.CustomUNet2DCM import CustomUNet2DConditionModel
from PIL import Image
from compel import Compel
import random
from pathlib import Path

def cutmix(image_1: Image.Image, image_2: Image.Image) -> Image.Image:
    """Apply CutMix augmentation to two images."""
    lam = 0.5
    W1, H1 = image_1.size
    W2, H2 = image_2.size
    
    crop_size = int(min(W2, H2) * np.sqrt(1 - lam))
    cx, cy = W2 // 2, H2 // 2
    bbx1, bby1 = cx - crop_size // 2, cy - crop_size // 2
    bbx2, bby2 = bbx1 + crop_size, bby1 + crop_size
    
    crop_area = image_2.crop((bbx1, bby1, bbx2, bby2))
    
    corner_margin = int(min(W1, H1) * 0.05)
    corners = [
        (corner_margin, corner_margin),
        (W1 - crop_size - corner_margin, corner_margin),
        (corner_margin, H1 - crop_size - corner_margin),
        (W1 - crop_size - corner_margin, H1 - crop_size - corner_margin)
    ]
    
    paste_x, paste_y = corners[np.random.randint(0, 4)]
    
    mixed_image = image_1.copy()
    mixed_image.paste(crop_area, (paste_x, paste_y))
    
    return mixed_image

def setup_pipeline(model_id: str, device: torch.device) -> CustomSDPipeline:
    """Set up the Stable Diffusion pipeline."""
    pipeline = CustomSDPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        variant="fp16",
        disable_progress_bar=True,
    )
    
    custom_unet = CustomUNet2DConditionModel(**pipeline.unet.config)
    custom_unet.load_state_dict(pipeline.unet.state_dict())
    pipeline.unet = custom_unet
    pipeline = pipeline.to(device, dtype=torch.float16)
    
    return pipeline

def main():
    # Fixed parameters
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    steps = 45
    guidance_scale = 8.0
    lora_dir = "/home/zhouyufan/Projects/DE/dataset_expansion/gen/finetuned/flower102"
    
    # Setup pipeline
    pipeline = setup_pipeline(model_id, device)
    lora_path = Path(lora_dir)
    
    # Example for two specific classes
    class_idx = 1  # First class
    rand_class = 2  # Second class
    
    # Load LoRA weights
    pipeline.load_lora_weights(lora_path / f"class{class_idx:03d}", adapter_name=f"class{class_idx:03d}")
    pipeline.load_lora_weights(lora_path / f"class{rand_class:03d}", adapter_name=f"class{rand_class:03d}")
    
    # Set adapters with custom weights
    scales_up = {"unet": {"down": 1.0, "mid": 0.5, "up": 0.0}}
    scales_down = {"unet": {"down": 0.0, "mid": 0.5, "up": 1.0}}
    pipeline.set_adapters([f"class{class_idx:03d}", f"class{rand_class:03d}"], [scales_up, scales_down])
    pipeline.fuse_lora(adapter_names=[f"class{class_idx:03d}", f"class{rand_class:03d}"])
    
    # Setup Compel processor
    compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
    
    # Example image paths (you'll need to set these)
    img1_path = "path/to/first/image.jpg"
    img2_path = "path/to/second/image.jpg"
    
    # Load and mix images
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    mix_image = cutmix(img1, img2)
    mix_image_tensor = pipeline.image_processor.preprocess(mix_image).to(device)
    
    # Generate prompts and embeddings
    prompt1 = f"A photo of a cls{class_idx:03d} flower"
    prompt2 = f"A photo of a cls{rand_class:03d} flower"
    encoder_embedding = compel_proc(prompt1)
    decoder_embedding = compel_proc(prompt2)
    
    # Generate the image
    image = pipeline(
        prompt_embeds=encoder_embedding,
        prompt_embeds_2=decoder_embedding,
        image=mix_image_tensor,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
    ).images[0]
    
    # Save the generated image
    output_dir = Path("generated_images")
    output_dir.mkdir(exist_ok=True)
    image.save(output_dir / f"mixed_class{class_idx:03d}_with_class{rand_class:03d}.jpg")
    
    # Cleanup
    pipeline.unload_lora_weights()

if __name__ == "__main__":
    main()


# nohup python hybrid.py > hybrid.log 2>&1 &