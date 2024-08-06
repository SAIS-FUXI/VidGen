#coding=utf-8
"""
Sample new images from a pre-trained DiT.
"""
import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import time
import json
import pickle
import random
import argparse
from omegaconf import OmegaConf
import cv2
from einops import rearrange
from PIL import Image
import numpy as np
import imageio
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
from contextlib import suppress

from core.diffusion import IDDPM
#from core.models.video_vae import CausualVAEVideo
from core.models.video_vae.causual_vae_video_patched import CausualVAEVideo
from core.models import model_cls
from core.models.embedder import FrozenCLIPEmbedder
from core.models import T5Encoder, ClipEncoder

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_autocast(precision, cache_enabled=True):
    if precision == "amp_bfloat16" or precision == "amp_bf16" or precision == 'bf16':
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16, cache_enabled=cache_enabled)
    elif precision == 'fp16':
        return lambda: torch.cuda.amp.autocast(dtype=torch.float16, cache_enabled=cache_enabled)
    elif precision == 'fp32':
        return suppress
    else:
        raise ValueError('not supported precision: {}'.format(precision))

class SimpleDataset(Dataset):
    def __init__(self, prompts, cond_images=None):
        if cond_images is not None:
            assert len(cond_images) == len(prompts)
        else:
            cond_images = [None] * len(prompts)

        self.prompts = []
        for i, (p, im) in enumerate(zip(prompts, cond_images)):
            self.prompts.append({'id': i, 'prompt': p, 'cond_image': im})

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return self.prompts[index]

def collate_fn(samples):
    return {
        'id': [s['id'] for s in samples],
        'prompt': [s['prompt'] for s in samples],
        'cond_image': [s['cond_image'] for s in samples]
    }

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--output_sample_dir", type=str)

    parser.add_argument('--sampling_algo', type=str, default="iddpm")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument('--cfg_scale', type=float, default=2.5)

    #parser.add_argument('--testset_name', type=str)
    parser.add_argument('--test_sample', type=str)
    parser.add_argument("--is_video", action="store_true")

    parser.add_argument("--resolution_height", type=int)
    parser.add_argument("--resolution_width", type=int)
    parser.add_argument("--num_slice_for_long_video", type=int)
    parser.add_argument("--seed", type=int)

    parser.add_argument('--long_video_method', type=str, default="slice", choices=["slice", "whole"])

    parser.add_argument("--pdb_debug", action="store_true")
    args = parser.parse_args()
    

    model_config = OmegaConf.load(args.model_config)
    args.precision = model_config.precision
    if args.seed is None:
        args.seed = model_config.seed
    if args.resolution_height is None:
        args.resolution_height = model_config.resolution_video
    if args.resolution_width is None:
        args.resolution_width = model_config.resolution_video
    args.mode_various_resolution = model_config.mode_various_resolution

    args.prob_self_condition = getattr(model_config.diffusion, 'prob_self_condition', 0)
    args.prob_text_condition = getattr(model_config.model, 'prob_text_condition', 1.0)
    
    if args.pdb_debug:
        import pdb; pdb.set_trace()

    random.seed(41)
    gt_latents = None

    if args.test_sample:
        with open(args.test_sample, "r") as f:
            datasets = json.load(f)
            #datas = json.load(open('datasets/text_to_video/UCF101/processed_ucf101_test.json'))
            args.prompts = []
            for d in datasets:
                if d['caption'] not in args.prompts:
                    args.prompts.append(d['caption'])
                #if d.strip() not in args.prompts:
            args.cond_image_file = None

    if args.pdb_debug:
        pass
    else:
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=int(os.getenv('WORLD_SIZE', '1')),
            rank=int(os.getenv('RANK', '0')),
        )
        torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda"

    if args.precision == 'bf16':
        dtype = torch.bfloat16
    elif args.precision == 'fp16':
        dtype = torch.float16
    elif args.precision == 'fp32':
        dtype = torch.float32

    # build vae model
    if model_config.vae.type == 'CausualVAEVideo':
        videovae_config = OmegaConf.load(model_config.vae.config)
        args.fps_ds = videovae_config.fps_ds
        videovae = CausualVAEVideo(ddconfig=videovae_config.ddconfig, embed_dim=videovae_config.embed_dim)
        checkpoint = torch.load(model_config.vae.from_pretrained)
        msg = videovae.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("vae load from {}".format(model_config.vae.from_pretrained))
        print(msg)
        for p in videovae.parameters():
            p.requires_grad = False
        t_downsampling = 2**(sum([s==2 for s in videovae_config.ddconfig.temporal_stride]))
        assert t_downsampling in [4, 8]
        videovae.patch_size = (t_downsampling, 8, 8)    # TODO: vae spatial downsampling is fixed to 8

        if args.is_video:
            #videovae.scaling_factor = getattr(videovae_config, f"scaling_factor_size_{model_config.resolution_video}_video") # 0.15100
            videovae.scaling_factor = getattr(videovae_config, f"scaling_factor_video")
        else:
            #videovae.scaling_factor = getattr(videovae_config, f"scaling_factor_size_{model_config.resolution_video}_image") # 0.15100
            videovae.scaling_factor = getattr(videovae_config, f"scaling_factor_image") 
        videovae.type = 'VAEVideo'

    else:
        raise ValueError()
    videovae = videovae.eval().to(device)
  
    args.num_frames_video = model_config.num_frames_video
    assert args.num_frames_video == videovae_config.ddconfig.t_frames, \
        "make sure that num_frames_video is set to t_frames of vae. For long video, increase the setting of num_slice_for_long_video"

    if args.num_slice_for_long_video is not None:
        num_slice_for_long_video = args.num_slice_for_long_video
    else:
        num_slice_for_long_video = getattr(model_config, 'num_slice_for_long_video', 1)
    assert num_slice_for_long_video >= 1
    if num_slice_for_long_video > 1:
        if args.long_video_method == 'slice':
            args.num_frames_video = args.num_frames_video * num_slice_for_long_video
        elif args.long_video_method == 'whole':
            args.num_frames_video = 1 + 16 * num_slice_for_long_video
        else:
            raise ValueError()
    
    input_size = [args.num_frames_video, args.resolution_height, args.resolution_width]

    if model_config.vae.type == 'CausualVAEVideo':
        for i in range(3):
            if i == 0:
                if args.long_video_method == 'slice':
                    assert (input_size[i] // num_slice_for_long_video - 1) % videovae.patch_size[i] == 0, "Input size must be divisible by patch size"
                    input_size[i] = (1 + (input_size[i] // num_slice_for_long_video - 1) // videovae.patch_size[i]) * num_slice_for_long_video
                elif args.long_video_method == 'whole':
                    assert (input_size[i] - 1) % videovae.patch_size[i] == 0, "Input size must be divisible by patch size"
                    input_size[i] = 1 + (input_size[i] - 1) // videovae.patch_size[i]
                else:
                    raise ValueError()
            else:
                assert input_size[i] % videovae.patch_size[i] == 0, "Input size must be divisible by patch size"
                input_size[i] = input_size[i] // videovae.patch_size[i]
    else:
        raise ValueError()
    
    if not args.is_video:
        input_size[0] = 1

    # build text enocder
    if model_config.text_encoder.type == 't5':
        text_encoder_config = dict(model_config.text_encoder)
        text_encoder_config.pop('type')
        text_encoder_config['shardformer'] = False
        text_encoder = T5Encoder(**text_encoder_config, device=device)    # T5 must be fp32
        for p in text_encoder.t5.model.parameters():
            p.requires_grad = False
        text_encoder.t5.model.eval().to(device)
    else:
        raise ValueError()

    # build model    
    model_kwargs = dict(model_config.model)
    model_kwargs.pop('type')
    model_kwargs['model_max_length'] = model_config.text_encoder.model_max_length
    model_kwargs['prob_self_condition'] = args.prob_self_condition
    if model_config.text_encoder.type == 'clip':
        model_kwargs['caption_channels'] = 768
    if not getattr(model_config.model, 'enable_rope', True):
        model_kwargs['input_size'] = input_size
    model = model_cls[model_config.model.type](**model_kwargs).to(dtype)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("model load from: {}".format(args.checkpoint))
    print(msg)
    model.eval().to(device)

    dataset = SimpleDataset(prompts=args.prompts, cond_images=args.cond_image_file)
    if args.pdb_debug:
        sampler = None 
    else:
        sampler = InferenceSampler(len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    for batch in dataloader:
        assert len(batch['id']) == len(batch['prompt']) == len(batch['cond_image']) == 1, 'batch should be 1'

        prompt = batch['prompt'][0]
        n = batch['id'][0]
        cond_image = batch['cond_image'][0]

        autocast = get_autocast(args.precision, cache_enabled=True)
        with torch.no_grad():
            model_kwargs = {}

            if args.prob_text_condition > 0:
                text_cond_encode = text_encoder.encode([prompt])
                y = text_cond_encode['y']                                                   # [1, 1, L, D]
                y_mask = text_cond_encode['mask']                                           # [1, L]
                null_y = model.y_embedder.y_embedding[None, None, :, :]                     # [1, 1, L, D]
            else:
                y = y_mask = None

            z = torch.randn(1, videovae.embed_dim, *input_size, device=device) # [1,8,9,64,64]

            if args.sampling_algo in ['iddpm', 'ddim']:
                z = torch.cat([z, z], 0)
                diffusion = IDDPM(str(args.num_sampling_steps),
                                  prob_self_condition=args.prob_self_condition)

                model_kwargs['cfg_scale'] = args.cfg_scale
                if args.prob_text_condition > 0:
                    model_kwargs['y'] = torch.cat([y, null_y])
                    if y_mask is not None:
                        model_kwargs['y_mask'] = torch.cat([y_mask, y_mask])
                else:
                    model_kwargs['y'] = model_kwargs['y_mask'] = None

                with autocast():
                    if args.sampling_algo == 'iddpm':
                        samples = diffusion.p_sample_loop(
                            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                            device=device
                        )
                    elif args.sampling_algo == 'ddim':
                        samples = diffusion.ddim_sample_loop(
                            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                            device=device
                        )
                    else:
                        raise ValueError()

                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            
            else:
                raise ValueError()

            samples = samples / videovae.scaling_factor
            B, _, T, H, W = samples.shape
            assert B == 1

            with autocast():
                if videovae.type == 'VAEVideo':
                    if (num_slice_for_long_video > 1) and (T > 1) and (args.long_video_method == 'slice'):
                        video_recons = torch.cat([videovae.decode(t) for t in torch.split(samples, T//num_slice_for_long_video, dim=2)], dim=2)
                    
                    elif (num_slice_for_long_video > 1) and (T > 1) and (args.long_video_method == 'whole'):
                        t_start = time.time()
                        video_recons = videovae.decode(samples)
                        t_end = time.time()
                    
                    else:
                        video_recons = videovae.decode(samples)
                else:
                    samples = rearrange(samples, 'b z t h w -> (b t) z h w')
                    video_recons = videovae.decode(samples)['sample']
                    video_recons = rearrange(video_recons, '(b t) c h w -> b c t h w', b=B)

            video_recons = torch.clamp(video_recons, -1.0, 1.0)
            video_recons = (video_recons + 1.0) / 2.0
            video_recons = (video_recons * 255).type(torch.uint8)

        print(f"video_recons: {video_recons.shape}")
        if T == 1: # image
            video_recons = rearrange(video_recons, 'b c t h w -> (b t) h w c').cpu().numpy()
            outfile = args.output_sample_dir + "/sampling_{}_{}_{}x{}_seed{}/t2i".format(args.sampling_algo, args.num_sampling_steps, args.resolution_height, args.resolution_width, args.seed) \
                        + '_{}.png'.format(n)
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            cv2.imwrite(outfile, video_recons[0][:, :, ::-1])
            print('write to: {}'.format(outfile))
        else:
            outfile = args.output_sample_dir + "/sampling_{}_{}_{}x{}_video_t-{}_seed{}/t2v".format(args.sampling_algo, args.num_sampling_steps, args.resolution_height, args.resolution_width, args.num_frames_video, args.seed) \
                        + '_{}.png'.format(n)
            os.makedirs(os.path.dirname(outfile), exist_ok=True)

            # if True:
            #     images = video_recons
            #     images = rearrange(images, "b c t h w -> b h (t w) c").cpu().numpy()
            #     cv2.imwrite(outfile, images[0][:, :, ::-1])
            #     print('write to: {}'.format(outfile))

            images = rearrange(video_recons, "b c t h w -> (b t) h w c").cpu().numpy()
            if True:
                gif = []
                for i in range(len(images)):
                    im = imageio.core.util.Array(images[i][:, :, :])
                    gif.append(im)
                imageio.mimsave(outfile.replace('.png', '.gif'), gif, 'GIF', quality=10, fps=args.fps_ds)
                print('write to: {}'.format(outfile.replace('.png', '.gif')))
                
            if True:
                num_repeat  = int(25 / args.fps_ds)
                _, H, W, _ = images.shape
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                out_file = outfile.replace('.png', '.mp4')
                out = cv2.VideoWriter(out_file, fourcc, 24, (W, H))
                for i in range(len(images)):
                    for _ in range(num_repeat):
                        out.write(images[i][:, :, ::-1])
                out.release()
                print('write to: {}'.format(out_file))

        print("prompt: {}".format(prompt))