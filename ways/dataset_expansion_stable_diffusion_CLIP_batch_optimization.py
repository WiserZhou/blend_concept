'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import os.path
import shutil
import time
import random

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable 
import models.cifar as models
import torchvision.models
import clip
from utils import AverageMeter
import PIL
from PIL import Image
 


###### Stable Diffusion ######
import core 
import math
from omegaconf import OmegaConf
from tqdm import tqdm
from itertools import islice
from einops import rearrange, repeat



from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
##################################


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])) 
model_names.append('resnet50')
model_names.append('CLIP-VIT-B32')
model_names.append('CLIP-VIT-L14')
parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--data_dir', default='data', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--data_save_dir', default='data', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--expanded_number', default=50000, type=int)
parser.add_argument('--expanded_number_per_sample', default=5, type=int)
parser.add_argument('--expanded_batch_size', default=2, type=int)
parser.add_argument('--constraint_value', default=0.1, type=float)
# Optimization options
parser.add_argument('--max_epochs', default=10000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=1, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[41, 81],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)') 
parser.add_argument('--total_split', default=8, type=int)
parser.add_argument('--split',  default=0, type=int, help='Dividing classes into 5 parts, the index of which parts') 

# Stable diffusion 
parser.add_argument(
    "--skip_grid",
    action='store_false',
    help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
)
 

parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)

parser.add_argument(
    "--plms",
    action='store_true',
    help="use plms sampling",
)
parser.add_argument(
    "--fixed_code",
    action='store_true',
    help="if enabled, uses the same starting code across all samples ",
)

parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=1,
    help="sample this often",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor, most often 8 or 16",
) 
parser.add_argument(
    "--n_rows",
    type=int,
    default=1,
    help="rows in the grid (default: n_samples)",
)
parser.add_argument(
    "--scale",
    type=float,
    default=50.0,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)

parser.add_argument(
    "--strength",
    type=float,
    default=0.5,
    help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
)
parser.add_argument(
    "--config",
    type=str,
    default="configs/stable-diffusion/v1-inference.yaml",
    help="path to config which constructs model",
)
parser.add_argument(
    "--ckpt",
    type=str,
    default="models/model.ckpt",
    help="path to checkpoint of model",
)
# RandAugment 
parser.add_argument('--noise_std', default=1, type=float)
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='CLIP-ViT-B32',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed') 
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES') 
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
 

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


class Custom_ImageFolder(data.Dataset): 
    def __init__(self, root, loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        rgb_mean = (0.4914, 0.4822, 0.4465)

        self.transform_train_normalized = transforms.Compose([
            transforms.Resize((512,512), interpolation=PIL.Image.BICUBIC),
            # transforms.RandomCrop(256), 
            # transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]) 
 
        self.transform_original_normalized = transforms.Compose([
                transforms.Resize((224,224), interpolation=PIL.Image.BICUBIC),  
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])  
    
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        augmented_img = self.transform_train_normalized(img)  
        original_img = self.transform_original_normalized(img)

        return  original_img, augmented_img, target, index
     
    def __len__(self):
        return len(self.imgs)

    
class MaskedSampler(torch.utils.data.Sampler):
    def __init__(self, indices, mask):
        self.indices = indices
        self.mask = mask

    def __iter__(self):
        return (self.indices[i] for i in torch.nonzero(self.mask))

    def __len__(self):
        return len(self.mask)  
 
 

def fix_seed(seed=888):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False


def load_model_from_config(config, ckpt, verbose=False): 
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    model.cuda()
    model.eval()
    return model

def tensor_clamp(t, min, max, in_place=True):
    if not in_place:
        res = t.clone()
    else:
        res = t
    idx = res.data < min
    res.data[idx] = min[idx]
    idx = res.data > max
    res.data[idx] = max[idx]

    return res

def linfball_proj(center, radius, t, in_place=True):
    return tensor_clamp(t, min=center - radius, max=center + radius, in_place=in_place)
 
    
def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
         os.makedirs(args.checkpoint)

    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    # Data  
    num_classes=102 
    total_trainset = Custom_ImageFolder(root=args.data_dir + '/train')   

    #class_names1 = total_trainset.classes
    with open("./cat_to_name.json",'r') as load_f:
         load_dict = json.load(load_f)
         #print(load_dict)
    class_name_index = total_trainset.classes 
    #class_name_index.sort(key=lambda d:int(d))
    #class_name_index1 = sorted(class_name_index,key=lambda d:int(d))
    #class_name_index_new = [str(x) for x in class_name_index1]
    class_names = [] 
    for i in range(num_classes):
        index = class_name_index[i]
        # Extract last 3 digits, convert to int, and add 1
        new_index = str(int(index[-3:]) + 1)
        # Pad with zeros to ensure 3 digits
        # new_index = new_index.zfill(3)
        class_names.append(load_dict[new_index])
    
    print(class_names)

    print("{} Mode: Contain {} images".format("Train", len(total_trainset)))


    #trainset1 = datasets.ImageFolder(args.data_dir) 
    total_data_number = len(total_trainset)   
    number_per_split = math.ceil(total_data_number/args.total_split)
    if args.split==(args.total_split-1) and total_data_number < number_per_split * (args.split + 1):
        mask = list(range(number_per_split * args.split, total_data_number))  
    else:
        mask = list(range(number_per_split * args.split, number_per_split * (args.split + 1))) 

    trainset = torch.utils.data.Subset(total_trainset, mask) 
 
    
    data_number = len(trainset)
    indices =  np.arange(0, data_number)
    expanded_number_per_sample = args.expanded_number_per_sample
    expanded_number = expanded_number_per_sample * data_number

    # Model
    print("==> creating model '{}'".format(args.arch))
    # create model
    #num_classes = 1000
    dim_feature = 2048    
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    baseWidth=args.base_width,
                    cardinality=args.cardinality,
                )
    elif args.arch == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True) 
        model.fc = nn.Linear(dim_feature, num_classes)
    elif args.arch == 'CLIP-VIT-B32':     
        model, preprocess = clip.load("ViT-B/32")       
        text_descriptions = [f"This is a photo of a {label}" for label in class_names]
        text_tokens = clip.tokenize(text_descriptions).cuda()
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_classifier = text_features
    elif args.arch == 'CLIP-VIT-L14':     
        model, preprocess = clip.load("ViT-L/14")       
        text_descriptions = [f"This is a photo of a {label}" for label in class_names]
        text_tokens = clip.tokenize(text_descriptions).cuda()
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_classifier = text_features        
        
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    
    model = model.cuda()      
    print('Model CLIP loaded.')
 
    device = torch.device('cuda')  
    config = OmegaConf.load(f"{args.config}")
    GIF_model   = load_model_from_config(config, f"{args.ckpt}")    
    GIF_model = GIF_model.eval().to(device)
    DDIM_sampler = DDIMSampler(GIF_model) 
    print('Model Stable diffusion loaded.')
  
    cudnn.benchmark = True
    print('Total params of CLIP: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0)) 
    print('Total params of Stable Diffusion: %.2fM' % (sum(p.numel() for p in GIF_model.parameters())/1000000.0))  

    # Train and val
    class_save_list = [0] * num_classes
    selected_class_save_list = [0] * num_classes
    sample_wise_save_list = torch.tensor([0] * data_number)
    epoch = 0  
    
    # Initialize tqdm progress bar
    pbar = tqdm(total=expanded_number, desc='Overall Progress', 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    while sum(sample_wise_save_list) < expanded_number:
        # print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.max_epochs, state['lr']))
        masks = (sample_wise_save_list < expanded_number_per_sample)
        sampler = MaskedSampler(indices, masks)
        if args.expanded_batch_size <= (expanded_number_per_sample - sample_wise_save_list[0]):
            batch_size = args.expanded_batch_size
        else:
            batch_size = expanded_number_per_sample - sample_wise_save_list[0]

        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, sampler=sampler, shuffle=False, num_workers=args.workers)
        sample_wise_save_list, class_save_list, selected_class_save_list = dataset_expansion(trainloader, model, GIF_model, DDIM_sampler, epoch, use_cuda, class_name_index, text_classifier, class_save_list, selected_class_save_list, sample_wise_save_list, expanded_number_per_sample, number_per_split, batch_size)
        
        # Update progress bar
        pbar.update(sum(x.item() if isinstance(x, torch.Tensor) else x for x in sample_wise_save_list) - pbar.n)
        
        epoch += 1
        pbar.set_postfix({'Epoch': epoch, 'Expanded': sum(class_save_list), 'Selected': sum(selected_class_save_list)})
        
        if epoch == args.max_epochs:
            break

    # Close progress bar
    pbar.close()

    print("Final expanded data size:", sum(class_save_list))
    print("Final selected expanded data size:", sum(selected_class_save_list))
    print("Final Sample_wise_save_list:", sample_wise_save_list)

def dataset_expansion(train_loader, model, GIF_model, DDIM_sampler, epoch, use_cuda, 
                     class_names, text_classifier, class_save_list, selected_class_save_list, 
                     sample_wise_save_list, expanded_number_per_sample, number_per_split, batch_size):
    """
    Expand dataset by generating new images using Stable Diffusion and CLIP guidance
    
    Args:
        train_loader: DataLoader containing original images
        model: CLIP model for feature extraction
        GIF_model: Stable Diffusion model
        DDIM_sampler: DDIM sampler for Stable Diffusion
        ...
    """
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter() 
    end = time.time()
    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    
    for batch_idx, (original_inputs, augmented_inputs, targets, index) in enumerate(train_loader):
        # Part 1: Initial Setup and Data Preparation
        with torch.no_grad():
            data_time.update(time.time() - end)
            if use_cuda:  
                original_inputs, augmented_inputs, targets = original_inputs.cuda(), augmented_inputs.cuda(), targets.cuda() 
            expanded_number_per_sample_each_epoch = int(batch_size)
            
            # Generate prompts for image generation
            domain_list = ["an image of", "a real-world photo of", "a cartoon image of", 
                         "an oil painting of", "a sketch of"]
            adjective_list = ["", "colorful", "stylized", "high-contrast", "low-contrast", 
                            "posterized", "solarized", "sheared", "bright", "dark"]
            prompts_list = []
            for i in range(expanded_number_per_sample_each_epoch):
                prompts_list.append("{} {} {}".format(
                    random.choice(domain_list), 
                    random.choice(adjective_list), 
                    class_names[targets]
                ))
            prompts = prompts_list

            # Part 2: Initial Image Processing
            # Prepare input image for Stable Diffusion
            augmented_inputs = augmented_inputs * 2 - 1 
            init_image = repeat(augmented_inputs, '1 ... -> b ...', b=expanded_number_per_sample_each_epoch)
            init_latent = GIF_model.get_first_stage_encoding(GIF_model.encode_first_stage(init_image))
            
            # Setup DDIM sampling parameters
            DDIM_sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=args.ddim_eta, verbose=False)
            t_enc = int(args.strength * args.ddim_steps)

            # Part 3: CLIP Feature Extraction
            # Get features and probabilities from original image
            image_original_features = model.encode_image(original_inputs).float().detach()
            image_original_features = image_original_features/image_original_features.norm(dim=-1, keepdim=True)
            original_outputs = (100.0 * image_original_features @ text_classifier.T)
            original_output_prob = original_outputs.softmax(dim=-1)
            original_output_index = original_output_prob.argmax(dim=1)
            original_output_entropy = -(original_output_prob * torch.log(original_output_prob)).sum(dim=1)

            # Part 4: Initial Image Generation with Stable Diffusion
            with GIF_model.ema_scope():
                all_samples = list()
                uc = None
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                if args.scale != 1.0:
                    uc = GIF_model.get_learned_conditioning(expanded_number_per_sample_each_epoch * [""])
                c = GIF_model.get_learned_conditioning(prompts)
                z_enc = DDIM_sampler.stochastic_encode(
                    init_latent, 
                    torch.tensor([t_enc]*expanded_number_per_sample_each_epoch).to('cuda')
                )
                samples = DDIM_sampler.decode(
                    z_enc, c, t_enc, 
                    unconditional_guidance_scale=args.scale,
                    unconditional_conditioning=uc,
                )

        # Part 5: Optimization Loop
        # Add noise to samples and optimize
        samples_temp = samples.clone().cuda()
        channel_noise_dim = samples_temp.shape[1]
        
        # Initialize noise variables
        channel_noise = Variable(
            torch.rand([expanded_number_per_sample_each_epoch,channel_noise_dim, 1,1]),  
            requires_grad=True
        ).cuda()
        channel_noise_bias = Variable(
            torch.zeros([expanded_number_per_sample_each_epoch,channel_noise_dim, 1,1]).data.normal_(0, 1),
            requires_grad=True
        ).cuda()
        
        # Apply noise and get features
        samples_temp = samples_temp * (1 + channel_noise) + channel_noise_bias
        x_samples = GIF_model.decode_first_stage(samples_temp)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        augmented_inputs1 = core.imresize(x_samples, sizes=(224, 224))
        
        # Get features and calculate scores
        image_augmented_features = model.encode_image(augmented_inputs1).float()
        image_augmented_features = image_augmented_features/image_augmented_features.norm(dim=-1, keepdim=True)
        augmented_output_prob = (100.0 * image_augmented_features @ text_classifier.T).softmax(dim=-1)
        augmented_output_entropy = -(augmented_output_prob * torch.log(augmented_output_prob)).sum(dim=1)

        # Calculate optimization score
        delta_entropy = augmented_output_entropy - original_output_entropy
        delta_entropy = delta_entropy.mean()
        original_output_index1 = original_output_index.unsqueeze(1)
        augmented_output_prob_top1 = torch.gather(augmented_output_prob,1,original_output_index1).mean()

        mean_image_augmented_features = torch.mean(image_augmented_features, 0, keepdim=True).repeat([expanded_number_per_sample_each_epoch,1])
        image_augmented_features = F.log_softmax(image_augmented_features)
        target_mean_image_augmented_features = F.log_softmax(mean_image_augmented_features)
        divergence = kl_loss(image_augmented_features, target_mean_image_augmented_features)

        Score = delta_entropy + augmented_output_prob_top1 + divergence

        # Update noise parameters
        (channel_noise_grad, channel_noise_bias_grad) = torch.autograd.grad(Score, [channel_noise, channel_noise_bias])
        channel_noise.data.add_(0.1 * torch.sign(channel_noise_grad))
        channel_noise_bias.data.add_(0.1 * torch.sign(channel_noise_bias_grad))

        # Clean up memory
        del samples_temp, x_samples, augmented_inputs1, image_augmented_features, augmented_output_prob
        del augmented_output_entropy, delta_entropy, augmented_output_prob_top1, divergence
        del channel_noise_grad, channel_noise_bias_grad

        # Part 6: Save Generated Images
        with torch.no_grad():
            samples_temp = samples.clone().cuda()
            samples = samples * (1 + channel_noise) + channel_noise_bias
            linfball_proj(samples_temp, args.constraint_value, samples, in_place=True)
            
            with GIF_model.ema_scope():
                x_samples = GIF_model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                # Save individual images
                for i, image in enumerate(x_samples):
                    path = args.data_save_dir + "/{}".format(class_names[targets])
                    if not os.path.isdir(path):
                        os.makedirs(path)
                    save_image(image, path + "/stable_{}_expanded_{}.png".format(
                        args.split, selected_class_save_list[targets]))
                    class_save_list[targets] += 1
                    selected_class_save_list[targets] += 1
                
                # Save grid of images
                all_samples.append(x_samples)
                if not args.skip_grid:
                    grid = torch.stack(all_samples, 0).squeeze(0)
                    grid = make_grid(grid, nrow=expanded_number_per_sample_each_epoch)
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(
                        path + "/stable_{}_expanded_{}_grid.png".format(args.split, batch_idx))

        # Update timing and progress tracking
        batch_time.update(time.time() - end)
        end = time.time()
        expended_indexes = index - number_per_split * args.split
        sample_wise_save_list[expended_indexes] += expanded_number_per_sample_each_epoch

    # Clean up GPU memory
    torch.cuda.empty_cache()
    return sample_wise_save_list, class_save_list, selected_class_save_list
 
def save_checkpoint(state, is_best, best_acc, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        print("The best performance:", best_acc)
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
 
def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()

 