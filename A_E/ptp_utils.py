import abc

import cv2
import numpy as np
import torch
from IPython.display import display
from PIL import Image
from typing import Union, Tuple, List

from diffusers.models.attention_processor import Attention as CrossAttention

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = True) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if display_image:
        display(pil_img)
    return pil_img


class AttendExciteCrossAttnProcessor:
    """注意力处理器类，用于处理和存储交叉注意力信息"""

    def __init__(self, attnstore, place_in_unet):
        """
        初始化注意力处理器
        :param attnstore: 用于存储注意力信息的对象
        :param place_in_unet: 标识处理器在U-Net中的位置（up/mid/down）
        """
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        """
        处理注意力计算的主要函数
        :param attn: 交叉注意力层
        :param hidden_states: 当前层的隐藏状态
        :param encoder_hidden_states: 编码器的隐藏状态（用于交叉注意力）
        :param attention_mask: 注意力掩码
        """
        # 获取输入张量的形状信息
        batch_size, sequence_length, _ = hidden_states.shape
        # 准备注意力掩码
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # 计算查询向量
        query = attn.to_q(hidden_states)

        # 判断是否为交叉注意力（如果有编码器隐藏状态则为交叉注意力）
        is_cross = encoder_hidden_states is not None
        # 如果没有提供编码器隐藏状态，则使用输入的隐藏状态
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        # 计算键和值向量
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # 将维度从"注意力头"形式转换为"批次"形式
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 计算注意力分数
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # 存储注意力概率分布
        self.attnstore(attention_probs, is_cross, self.place_in_unet)

        # 使用注意力权重与值向量相乘
        hidden_states = torch.bmm(attention_probs, value)
        # 将维度转换回原始形式
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # 应用线性投影层
        hidden_states = attn.to_out[0](hidden_states)
        # 应用dropout层
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def register_attention_control(model, controller):

    attn_procs = {}
    cross_att_count = 0
    for name in model.unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else model.unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = AttendExciteCrossAttnProcessor(attnstore=controller, 
                                                          place_in_unet=place_in_unet)

    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def __init__(self, save_global_store=False):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super().__init__()
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        self.curr_step_index = 0


def aggregate_attention(attention_store: AttentionStore,
                        res: int,
                        from_where: List[str],
                        is_cross: bool,
                        select: int) -> torch.Tensor:
    """
    聚合不同层和注意力头在指定分辨率下的注意力图
    
    参数:
        attention_store: 存储了所有注意力图的AttentionStore对象
        res: 注意力图的目标分辨率（边长）
        from_where: 需要聚合的位置列表（例如：['up', 'down', 'mid']）
        is_cross: 是否处理交叉注意力（True）或自注意力（False）
        select: 选择处理的批次索引
    
    返回:
        torch.Tensor: 聚合后的注意力图
    """
    # 存储所有符合条件的注意力图
    out = []
    # 获取平均注意力图
    attention_maps = attention_store.get_average_attention()
    # 计算目标分辨率下的像素总数
    num_pixels = res ** 2
    
    # 遍历指定位置
    for location in from_where:
        # 构建键名并获取对应的注意力图列表
        key = f"{location}_{'cross' if is_cross else 'self'}"
        # 遍历该位置的所有注意力图
        for item in attention_maps[key]:
            # 只处理与目标分辨率匹配的注意力图
            if item.shape[1] == num_pixels:
                # 重塑注意力图维度并选择指定批次
                # 从 [batch, pixels, tokens] 转换为 [batch, heads, height, width, tokens]
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    
    # 在第一维度（头维度）上拼接所有注意力图
    out = torch.cat(out, dim=0)
    # 计算所有注意力头的平均值，得到最终的注意力图
    out = out.sum(0) / out.shape[0]
    return out
