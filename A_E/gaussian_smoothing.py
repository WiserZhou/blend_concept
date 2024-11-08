import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


class GaussianSmoothing(nn.Module):
    """
    对1d、2d或3d张量应用高斯平滑。
    使用深度可分离卷积对输入的每个通道分别进行滤波处理。
    
    参数说明:
        channels: 输入张量的通道数，输出将保持相同的通道数
        kernel_size: 高斯核的大小
        sigma: 高斯核的标准差
        dim: 数据的维度，默认为2（空间维度）
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        # 如果kernel_size是单个数字，则扩展为dim维的列表
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        # 如果sigma是单个数字，则扩展为dim维的列表
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # 生成高斯核：最终的核是每个维度高斯函数的乘积
        kernel = 1
        # 创建网格点坐标，用于计算高斯核
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])

        # 对每个维度计算高斯分布
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2  # 计算核的中心点
            # 计算高斯函数：f(x) = (1/σ√(2π)) * e^(-(x-μ)²/2σ²)
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # 归一化核，确保所有值的和为1
        kernel = kernel / torch.sum(kernel)

        # 重塑核的形状以用于深度可分离卷积
        # 添加两个维度：第一个用于输出通道，第二个用于输入通道
        kernel = kernel.view(1, 1, *kernel.size())
        # 在通道维度上重复核，使其匹配输入通道数
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        # 注册核为缓冲区（不会被优化器更新）
        self.register_buffer('weight', kernel)
        self.groups = channels  # 设置分组卷积的组数，等于通道数

        # 根据维度选择适当的卷积操作
        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        对输入应用高斯滤波
        
        参数:
            input: 需要应用高斯滤波的输入张量
            
        返回:
            filtered: 经过滤波后的输出张量
        """
        # 执行分组卷积操作，确保权重的dtype与输入一致
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)


class AverageSmoothing(nn.Module):
    """
    Apply average smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the average kernel.
        sigma (float, sequence): Standard deviation of the rage kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, dim=2):
        super(AverageSmoothing, self).__init__()

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = torch.ones(size=(kernel_size, kernel_size)) / (kernel_size * kernel_size)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply average filter to input.
        Arguments:
            input (torch.Tensor): Input to apply average filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
