from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from diffusers.utils import logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from A_E.gaussian_smoothing import GaussianSmoothing
from A_E.ptp_utils import AttentionStore, aggregate_attention

logger = logging.get_logger(__name__)

class AttendAndExcitePipeline(StableDiffusionPipeline):

    _optional_components = ["safety_checker", "feature_extractor"]

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return text_inputs, prompt_embeds

    def _compute_max_attention_per_index(self,
                                         attention_maps: torch.Tensor,
                                         indices_to_alter: List[int],
                                         smooth_attentions: bool = False,
                                         sigma: float = 0.5,
                                         kernel_size: int = 3,
                                         normalize_eot: bool = False) -> List[torch.Tensor]:
        """
        计算每个需要修改的token的最大注意力值
        
        参数:
            attention_maps: 注意力图张量, 形状为 [batch_size, height*width, seq_length]
            indices_to_alter: 需要修改的token索引列表
            smooth_attentions: 是否对注意力图进行高斯平滑
            sigma: 高斯平滑的标准差
            kernel_size: 高斯核的大小
            normalize_eot: 是否对EOT(end of text)token进行归一化处理
        
        返回:
            List[torch.Tensor]: 每个token对应的最大注意力值列表
        """
        # 初始化最后一个token的索引
        last_idx = -1
        
        # 如果需要EOT归一化，获取prompt的最后一个token索引
        if normalize_eot:
            prompt = self.prompt
            # 如果prompt是列表，取第一个元素
            if isinstance(self.prompt, list):
                prompt = self.prompt[0]
            # 获取prompt的token长度
            last_idx = len(self.tokenizer(prompt)['input_ids']) - 1
        
        # 提取文本相关的注意力图，去除第一个token(通常是[CLS])
        # 如果启用EOT归一化，则只保留到last_idx的部分
        attention_for_text = attention_maps[:, :, 1:last_idx]
        
        # 将注意力值放大100倍，使差异更明显
        attention_for_text *= 100
        
        # 对注意力值进行softmax归一化，使其和为1
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # 由于移除了第一个token，需要将索引值都减1
        indices_to_alter = [index - 1 for index in indices_to_alter]

        # 提取每个token的最大注意力值
        max_indices_list = []
        for i in indices_to_alter:
            # 获取当前token的注意力图
            image = attention_for_text[:, :, i]
            
            # 如果启用了注意力平滑
            if smooth_attentions:
                # 创建高斯平滑层
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                # 添加通道维度并进行边缘填充
                input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                # 应用高斯平滑并移除额外的维度
                image = smoothing(input).squeeze(0).squeeze(0)
                
            # 将当前token的最大注意力值添加到列表中
            max_indices_list.append(image.max())
            
        return max_indices_list

    def _aggregate_and_get_max_attention_per_token(self, attention_store: AttentionStore,
                                                   indices_to_alter: List[int],
                                                   attention_res: int = 16,
                                                   smooth_attentions: bool = False,
                                                   sigma: float = 0.5,
                                                   kernel_size: int = 3,
                                                   normalize_eot: bool = False):
        """
        聚合每个token的注意力并计算每个需要修改的token的最大激活值
        
        参数:
            attention_store: AttentionStore对象，存储了模型各层的注意力图
            indices_to_alter: 需要修改的token索引列表
            attention_res: 注意力图的分辨率，默认16x16
            smooth_attentions: 是否对注意力图进行平滑处理
            sigma: 高斯平滑的标准差，仅在smooth_attentions=True时使用
            kernel_size: 高斯核的大小，仅在smooth_attentions=True时使用
            normalize_eot: 是否对EOT(End Of Text)token进行归一化处理
        
        返回:
            max_attention_per_index: 每个待修改token的最大注意力值
        """
        
        # 聚合注意力图
        # aggregate_attention函数从attention_store中提取并合并指定层("up","down","mid")的注意力
        # res: 设置输出注意力图的分辨率
        # from_where: 指定要从哪些层提取注意力("up","down","mid")
        # is_cross: True表示使用交叉注意力
        # select: 选择第一个batch样本(0)
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0)
        
        # 计算每个待修改token的最大注意力值
        # _compute_max_attention_per_index函数处理聚合后的注意力图
        # 如果启用smooth_attentions，会使用高斯模糊对注意力图进行平滑处理
        # normalize_eot控制是否对EOT token进行特殊的归一化处理
        max_attention_per_index = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
            indices_to_alter=indices_to_alter,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot)
        
        return max_attention_per_index

    @staticmethod
    def _compute_loss(max_attention_per_index: List[torch.Tensor], return_losses: bool = False) -> torch.Tensor:
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        losses = [max(0, 1. - curr_max) for curr_max in max_attention_per_index]
        loss = max(losses)
        if return_losses:
            return loss, losses
        else:
            return loss

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents

    def _perform_iterative_refinement_step(self,
                                           latents: torch.Tensor,
                                           indices_to_alter: List[int],
                                           loss: torch.Tensor,
                                           threshold: float,
                                           text_embeddings: torch.Tensor,
                                           text_input,
                                           attention_store: AttentionStore,
                                           step_size: float,
                                           t: int,
                                           attention_res: int = 16,
                                           smooth_attentions: bool = True,
                                           sigma: float = 0.5,
                                           kernel_size: int = 3,
                                           max_refinement_steps: int = 20,
                                           normalize_eot: bool = False):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code according to our loss objective until the given threshold is reached for all tokens.
        """
        iteration = 0
        target_loss = max(0, 1. - threshold)
        while loss > target_loss:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            # noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
            self.unet.zero_grad()

            # Get max activation value for each subject token
            max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
                attention_store=attention_store,
                indices_to_alter=indices_to_alter,
                attention_res=attention_res,
                smooth_attentions=smooth_attentions,
                sigma=sigma,
                kernel_size=kernel_size,
                normalize_eot=normalize_eot
                )

            loss, losses = self._compute_loss(max_attention_per_index, return_losses=True)

            if loss != 0:
                latents = self._update_latent(latents, loss, step_size)

            # with torch.no_grad():
            #     noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample
            #     noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample

            try:
                # 将losses列表中的每个元素转换为Python数值
                # 如果元素是tensor则调用.item()转换
                # 如果元素已经是int则保持不变
                # 然后用np.argmax找出最大值的索引
                low_token = np.argmax([l.item() if type(l) != int else l for l in losses])
            except Exception as e:
                print(e)  # 打印异常信息
                # 如果上面的方法失败，直接对losses使用np.argmax
                low_token = np.argmax(losses)

            # 使用tokenizer解码得到对应的单词
            # text_input.input_ids[0] - 获取第一个样本的token ids
            # indices_to_alter[low_token] - 获取注意力最低的token在序列中的位置
            # tokenizer.decode - 将token id转换回单词
            low_word = self.tokenizer.decode(text_input.input_ids[0][indices_to_alter[low_token]])
            print(f'\t Try {iteration}. {low_word} has a max attention of {max_attention_per_index[low_token]}')

            if iteration >= max_refinement_steps:
                print(f'\t Exceeded max number of iterations ({max_refinement_steps})! '
                      f'Finished with a max attention of {max_attention_per_index[low_token]}')
                break

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        # noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
        self.unet.zero_grad()

        # Get max activation value for each subject token
        max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
            attention_store=attention_store,
            indices_to_alter=indices_to_alter,
            attention_res=attention_res,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot)
        loss, losses = self._compute_loss(max_attention_per_index, return_losses=True)
        print(f"\t Finished with loss of: {loss}")
        return loss, latents, max_attention_per_index

    @torch.no_grad()
    def __call__(
            self,
            # 基本参数
            prompt: Union[str, List[str]],                    # 文本提示词
            attention_store: AttentionStore,                  # 存储注意力图的对象
            indices_to_alter: List[int],                      # 需要修改的token索引列表
            attention_res: int = 16,                          # 注意力图的分辨率
            height: Optional[int] = None,                     # 输出图像高度
            width: Optional[int] = None,                      # 输出图像宽度
            
            # 扩散模型相关参数
            num_inference_steps: int = 50,                    # 推理步数
            guidance_scale: float = 7.5,                      # 分类器引导比例
            negative_prompt: Optional[Union[str, List[str]]] = None,  # 负面提示词
            num_images_per_prompt: Optional[int] = 1,         # 每个提示词生成的图像数量
            eta: float = 0.0,                                # 噪声调度器参数
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 随机数生成器
            
            # 潜在空间相关参数
            latents: Optional[torch.FloatTensor] = None,      # 初始潜在向量
            prompt_embeds: Optional[torch.FloatTensor] = None,  # 提示词嵌入
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,  # 负面提示词嵌入
            
            # 输出相关参数
            output_type: Optional[str] = "pil",               # 输出类型
            return_dict: bool = True,                         # 是否返回字典格式
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,  # 回调函数
            callback_steps: Optional[int] = 1,                # 回调步数
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,  # 交叉注意力参数
            
            # Attend-and-Excite特定参数
            max_iter_to_alter: Optional[int] = 25,            # 最大迭代修改次数
            run_standard_sd: bool = False,                    # 是否运行标准SD
            thresholds: Optional[dict] = {0: 0.05, 10: 0.5, 20: 0.8},  # 不同步骤的阈值
            scale_factor: int = 20,                          # 缩放因子
            scale_range: Tuple[float, float] = (1., 0.5),    # 缩放范围
            smooth_attentions: bool = True,                   # 是否平滑注意力图
            sigma: float = 0.5,                              # 高斯平滑的sigma值
            kernel_size: int = 3,                            # 高斯核大小
            sd_2_1: bool = False,                            # 是否使用SD 2.1版本
    ):
        """
        Pipeline的主要调用函数，用于生成图像
        """
        # 1. 设置图像尺寸
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. 验证输入参数
        self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 3. 设置批处理大小
        self.prompt = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 4. 设置设备和分类器引导
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 5. 编码提示词
        text_inputs, prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 6. 准备时间步
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 7. 准备潜在向量
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 8. 准备额外步骤参数
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. 设置缩放范围
        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))

        # 10. 设置最大迭代次数
        if max_iter_to_alter is None:
            max_iter_to_alter = len(self.scheduler.timesteps) + 1

        # 11. 开始去噪循环
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                with torch.enable_grad():
                    # 克隆latents并设置requires_grad=True以便计算梯度
                    latents = latents.clone().detach().requires_grad_(True)

                    # 使用UNet模型对latents进行前向传播,生成噪声预测
                    noise_pred_text = self.unet(latents, t, 
                                              encoder_hidden_states=prompt_embeds[1].unsqueeze(0),
                                              cross_attention_kwargs=cross_attention_kwargs).sample
                    # 清除UNet的梯度
                    self.unet.zero_grad()

                    # 获取每个需要修改的token的最大注意力值
                    max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
                        attention_store=attention_store,  # 存储注意力图的对象
                        indices_to_alter=indices_to_alter,  # 需要修改的token索引列表
                        attention_res=attention_res,  # 注意力图的分辨率
                        smooth_attentions=smooth_attentions,  # 是否平滑注意力图
                        sigma=sigma,  # 高斯平滑的sigma参数
                        kernel_size=kernel_size,  # 高斯核大小
                        normalize_eot=sd_2_1  # 是否归一化EOT token
                    )

                    if not run_standard_sd:
                        # 如果不是运行标准的Stable Diffusion
                        # 计算注意力损失
                        loss = self._compute_loss(max_attention_per_index=max_attention_per_index)

                        # 如果当前步骤在thresholds中且损失大于阈值,执行迭代优化
                        if i in thresholds.keys() and loss > 1. - thresholds[i]:
                            # 释放显存
                            del noise_pred_text
                            torch.cuda.empty_cache()
                            
                            # 执行迭代优化步骤,返回新的loss、latents和注意力值
                            loss, latents, max_attention_per_index = self._perform_iterative_refinement_step(
                                latents=latents,
                                indices_to_alter=indices_to_alter,
                                loss=loss,
                                threshold=thresholds[i],
                                text_embeddings=prompt_embeds,
                                text_input=text_inputs,
                                attention_store=attention_store,
                                step_size=scale_factor * np.sqrt(scale_range[i]),  # 动态步长
                                t=t,
                                attention_res=attention_res,
                                smooth_attentions=smooth_attentions,
                                sigma=sigma,
                                kernel_size=kernel_size,
                                normalize_eot=sd_2_1
                            )

                        # 如果当前步骤小于最大迭代次数,执行梯度更新
                        if i < max_iter_to_alter:
                            loss = self._compute_loss(max_attention_per_index=max_attention_per_index)
                            if loss != 0:
                                # 根据损失更新latents
                                latents = self._update_latent(
                                    latents=latents, 
                                    loss=loss,
                                    step_size=scale_factor * np.sqrt(scale_range[i])
                                )
                            print(f'Iteration {i} | Loss: {loss:0.4f}')

                # 如果使用分类器引导,将latents复制一份
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                # 根据当前时间步缩放latent输入
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # 使用UNet预测噪声残差
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # 如果使用分类器引导,合并无条件和有条件的预测结果
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # 使用scheduler计算前一个时间步的采样结果
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # 更新进度条和执行回调
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 12. 后处理
        image = self.decode_latents(latents)

        # 13. 运行安全检查器
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 14. 转换为PIL图像（如果需要）
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        # 15. 返回结果
        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
