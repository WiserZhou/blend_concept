from dataclasses import dataclass
from typing import Optional
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, logging
import torch
from torchvision import transforms
from tqdm.auto import tqdm
from PIL import Image

logging.set_verbosity_error()

@dataclass
class MagicMixConfig:
    """Configuration for MagicMix generation"""
    kmin: float = 0.3
    kmax: float = 0.6
    v: float = 0.5
    seed: int = 42
    steps: int = 50
    guidance_scale: float = 7.5
    device: str = "cuda:6" if torch.cuda.is_available() else "cpu"

class MagicMix:
    def __init__(self, config: Optional[MagicMixConfig] = None):
        """Initialize MagicMix with models and configuration"""
        self.config = config or MagicMixConfig()
        self.device = self.config.device
        
        # Initialize models
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(self.device)
        
        # Initialize scheduler
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            set_alpha_to_one=False,
        )

    def encode(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL image to latents"""
        with torch.no_grad():
            latent = self.vae.encode(transforms.ToTensor()(img).unsqueeze(0).to(self.device) * 2 - 1)
            latent = 0.18215 * latent.latent_dist.sample()
        return latent

    def decode(self, latent: torch.Tensor) -> Image.Image:
        """Convert latents to PIL image"""
        latent = (1 / 0.18215) * latent
        
        with torch.no_grad():
            img = self.vae.decode(latent).sample
        
        img = (img / 2 + 0.5).clamp(0, 1)
        img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
        img = (img * 255).round().astype("uint8")
        return Image.fromarray(img[0])

    def prep_text(self, prompt: str) -> torch.Tensor:
        """Convert prompt into text embeddings and unconditional embeddings"""
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embedding = self.text_encoder(text_input.input_ids.to(self.device))[0]
        
        uncond_input = self.tokenizer(
            "",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embedding = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        return torch.cat([uncond_embedding, text_embedding])

    def __call__(self, img: Image.Image, prompt: str) -> Image.Image:
        """Generate image using MagicMix algorithm"""
        tmin = self.config.steps - int(self.config.kmin * self.config.steps)
        tmax = self.config.steps - int(self.config.kmax * self.config.steps)

        text_embeddings = self.prep_text(prompt)
        self.scheduler.set_timesteps(self.config.steps)

        width, height = img.size
        encoded = self.encode(img)

        torch.manual_seed(self.config.seed)
        noise = torch.randn((1, self.unet.in_channels, height // 8, width // 8)).to(self.device)
        latents = self.scheduler.add_noise(encoded, noise, timesteps=self.scheduler.timesteps[tmax])

        # Main generation loop
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            if i > tmax:
                if i < tmin:  # layout generation phase
                    orig_latents = self.scheduler.add_noise(encoded, noise, timesteps=t)
                    input = (self.config.v * latents) + (1 - self.config.v) * orig_latents
                else:  # content generation phase
                    input = latents
                
                input = torch.cat([input] * 2)
                input = self.scheduler.scale_model_input(input, t)

                with torch.no_grad():
                    pred = self.unet(input, t, encoder_hidden_states=text_embeddings).sample

                pred_uncond, pred_text = pred.chunk(2)
                pred = pred_uncond + self.config.guidance_scale * (pred_text - pred_uncond)
                latents = self.scheduler.step(pred, t, latents).prev_sample

        return self.decode(latents)

def main():
    """CLI interface"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("img_file", type=str, help="Input image file")
    parser.add_argument("prompt", type=str, help="Text prompt")
    parser.add_argument("out_file", type=str, help="Output file")
    parser.add_argument("--kmin", type=float, default=0.3)
    parser.add_argument("--kmax", type=float, default=0.6)
    parser.add_argument("--v", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    
    args = parser.parse_args()
    config = MagicMixConfig(**vars(args))
    
    magic_mix = MagicMix(config)
    img = Image.open(args.img_file)
    out_img = magic_mix(img, args.prompt)
    out_img.save(args.out_file)

if __name__ == "__main__":
    main()
