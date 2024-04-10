# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import numpy as np
import cv2
import random

from utils import ImageNetandMask, RandomResizedCropAndFlip

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

from torchvision.utils import make_grid
from torchvision.transforms import Grayscale

import wandb


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

#################################################################################
#                                  Dataset Setup                                #
#################################################################################

class EncodedImageNetandMask(Dataset):
    def __init__(self, encoded_image_dir, encoded_mask_dir):
        self.encoded_image_dir = encoded_image_dir
        self.encoded_mask_dir = encoded_mask_dir
        self.samples = []  # A list of tuples (encoded_image_path, encoded_mask_path, class_index)
        self.class_names = []  # Assuming same structure for both encoded_image_dir and encoded_mask_dir

        for class_name in sorted(os.listdir(encoded_image_dir)):
            encoded_image_class_dir = os.path.join(encoded_image_dir, class_name)
            encoded_mask_class_dir = os.path.join(encoded_mask_dir, class_name)
            if not os.path.isdir(encoded_image_class_dir) or not os.path.isdir(encoded_mask_class_dir):
                continue
            self.class_names.append(class_name)
            class_index = self.class_names.index(class_name)

            for encoded_image_name in os.listdir(encoded_image_class_dir):
                if encoded_image_name.endswith('.npy'):
                    encoded_image_path = os.path.join(encoded_image_class_dir, encoded_image_name)
                    encoded_mask_name = encoded_image_name
                    encoded_mask_path = os.path.join(encoded_mask_class_dir, encoded_mask_name)
                    if os.path.exists(encoded_mask_path):
                        self.samples.append((encoded_image_path, encoded_mask_path, class_index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        encoded_image_path, encoded_mask_path, class_index = self.samples[idx]
        encoded_image = np.load(encoded_image_path)
        encoded_mask = np.load(encoded_mask_path)
        return (encoded_image, encoded_mask), class_index
    
#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    dataset = EncodedImageNetandMask(encoded_image_dir=args.data_path, encoded_mask_dir=args.mask_path)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        # collate_fn=custom_collate
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    #wandb setup
    wandb.init(project="DiT-condition-noise", config=args)
    wandb.watch(model)

    #Setup the val images 
    transform_val = RandomResizedCropAndFlip(input_size=256)
    dataset_val = ImageNetandMask(os.path.join('/home/jqi/Github/Data/imagenet_100/imagenet100', 'val'), os.path.join('/home/jqi/Github/Data/imagenet_100/imagenet100_masks', 'val'), transform=transform_val)
    loader_val = DataLoader(dataset_val, batch_size=4, shuffle=True)
    mask_transform = Grayscale(num_output_channels=1)


    for i, ((images, masks), class_indices) in enumerate(loader_val):
        val_images = images.to(device)
        val_masks = masks.to(device)
        
        # Create a grid of 4 images for comparison
        image_grid = make_grid(val_images, nrow=4, normalize=True, value_range=(-1, 1))
        mask_grid = make_grid(val_masks, nrow=4, normalize=True, value_range=(-1, 1))
        image_grid = image_grid.permute(1, 2, 0).cpu().numpy()
        mask_grid = mask_transform(mask_grid).permute(1, 2, 0).cpu().numpy()

        # Log the image and mask grids to wandb
        wandb.log({
            "val_masks": wandb.Image(mask_grid),
            "val_images": wandb.Image(image_grid)
        })
        break
    del loader_val
    del dataset_val
    del transform_val

    #Create latent masks
    z = vae.encode(val_masks).latent_dist.sample().mul_(0.18215)
    z = torch.cat([z, z], 0)
    y = torch.tensor(class_indices, device=device)
    y_null = torch.tensor([100] * 4, device=device)
    y = torch.cat([y, y_null], 0)
    val_model_kwargs = dict(y=y, cfg_scale=4.0)



    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    
    

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for (encoded_images, encoded_masks), class_indices in loader:
            encoded_images = encoded_images.to(device)
            encoded_masks = encoded_masks.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (encoded_images.shape[0],), device=device)
            model_kwargs = dict(y= class_indices)
            # model_kwargs = None
            #doing y, x since I want to use image to predict segmentation mask
            loss_dict = diffusion.training_losses(model, encoded_images, encoded_masks, t, model_kwargs) #added y as input to the model
            loss = loss_dict["loss"].mean()
            
            #update loss with acculated gradients
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if train_steps % args.gradient_accumulation_steps == 0:
                opt.step()
                opt.zero_grad()
                update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                #wandb logging
                wandb.log({"train_loss": avg_loss, "steps_per_sec": steps_per_sec}, step=train_steps)
            
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

        # Save DiT checkpoint (changed to epochs):
        if epoch % args.ckpt_every == 0 and epoch > 0:
            if rank == 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            dist.barrier()
        
        #Validation
        if epoch % args.val_epoch == 0 and train_steps > 0:
            # model.eval()
            with torch.no_grad():
                samples = diffusion.p_sample_loop(
                    ema.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=val_model_kwargs, progress=True, device=device
                )
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                samples = vae.decode(samples / 0.18215).sample
                sample_grid = make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
                sample_grid = sample_grid.permute(1, 2, 0).cpu().numpy()
                wandb.log({"val_samples": wandb.Image(sample_grid)}, step=train_steps)
            # model.train()


    # Save final DiT checkpoint:
    if rank == 0:
        checkpoint = {
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args
        }
        checkpoint_path = f"{checkpoint_dir}/final.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved final checkpoint to {checkpoint_path}")


    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...


    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--mask-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--val-epoch", type=int, default=50)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    args = parser.parse_args()
    main(args)

#tmux command
#torchrun --nnodes=1 --nproc_per_node=1 train_optimized.py --model DiT-S/4 --num-classes 100 --global-batch-size 256 --epochs 1400 --gradient-accumulation-steps 4 --num-workers 32 --ckpt-every 100 --data-path /home/jqi/Github/Data/imagenet_100/encoded_imagenet100 --mask-path /home/jqi/Github/Data/imagenet_100/encoded_imagenet100_masks > output.log 2>&1
    
#nohup torchrun --nnodes=1 --nproc_per_node=1 train.py --model DiT-S/8 --num-classes 1 --global-batch-size 32 --epochs 20 --ckpt-every 10 --data-path /home/jqi/Github/Data/Data/coco > output.log 2>&1 &
