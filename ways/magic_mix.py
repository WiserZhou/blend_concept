from dataclasses import dataclass
from typing import Optional
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, logging
import torch
from torchvision import transforms
from tqdm.auto import tqdm
from PIL import Image
import math

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
    text_feature_type: str = "pooled"  # ['pooled', 'mean', 'attention', 'max', 'multihead']
    num_heads: int = 8  # 用于multihead方式

class MultiHeadPooling(torch.nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.attention = torch.nn.MultiheadAttention(hidden_size, num_heads)
        
    def forward(self, x):
        x = x.transpose(0, 1)
        query = x.mean(dim=0, keepdim=True)
        out, _ = self.attention(query, x, x)
        return out.squeeze(0)

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

        # 如果使用multihead方式，初始化pooling层
        if self.config.text_feature_type == "multihead":
            self.pooling = MultiHeadPooling(
                self.text_encoder.config.hidden_size, 
                self.config.num_heads
            ).to(self.device)

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

    def _process_hidden_states(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process hidden states based on configured method"""
        if self.config.text_feature_type == "pooled":
            # 使用原始的pooled output
            return hidden_states

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)

        if self.config.text_feature_type == "mean":
            # 平均池化
            if attention_mask is not None:
                # 计算非填充位置的平均值
                return (hidden_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True))
            return hidden_states.mean(dim=1)

        elif self.config.text_feature_type == "max":
            # 最大池化
            if attention_mask is not None:
                hidden_states[~attention_mask.bool()] = float('-inf')
            return hidden_states.max(dim=1)[0]

        elif self.config.text_feature_type == "attention":
            # 注意力加权平均
            attention_weights = torch.nn.functional.softmax(
                (hidden_states @ hidden_states.transpose(-2, -1)) / math.sqrt(hidden_states.size(-1)),
                dim=-1
            )
            if attention_mask is not None:
                attention_weights = attention_weights * attention_mask.unsqueeze(1)
                attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            return (attention_weights @ hidden_states).mean(dim=1)

        elif self.config.text_feature_type == "multihead":
            # 多头注意力池化
            return self.pooling(hidden_states)

        raise ValueError(f"Unknown text_feature_type: {self.config.text_feature_type}")

    def prep_text(self, prompt: str) -> torch.Tensor:
        """Convert prompt into text embeddings using specified method"""
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        if self.config.text_feature_type == "pooled":
            text_embedding = self.text_encoder(text_input.input_ids.to(self.device))[1]
        else:
            hidden_states = self.text_encoder(text_input.input_ids.to(self.device))[0]
            attention_mask = text_input.attention_mask.to(self.device)
            text_embedding = self._process_hidden_states(hidden_states, attention_mask)

        # 处理无条件嵌入
        uncond_input = self.tokenizer(
            "",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        if self.config.text_feature_type == "pooled":
            uncond_embedding = self.text_encoder(uncond_input.input_ids.to(self.device))[1]
        else:
            uncond_states = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            uncond_mask = uncond_input.attention_mask.to(self.device)
            uncond_embedding = self._process_hidden_states(uncond_states, uncond_mask)
        
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
    parser.add_argument(
        "--text_feature_type",
        type=str,
        default="pooled",
        choices=["pooled", "mean", "attention", "max", "multihead"],
        help="Method to process text features"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads for multihead pooling"
    )
    
    args = parser.parse_args()
    config = MagicMixConfig(**vars(args))
    
    magic_mix = MagicMix(config)
    img = Image.open(args.img_file)
    out_img = magic_mix(img, args.prompt)
    out_img.save(args.out_file)

if __name__ == "__main__":
    main()
