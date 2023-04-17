import os
import torch
import argparse
import itertools
import numpy as np
from unet import Unet
from tqdm import tqdm
import torch.optim as optim
from diffusion import GaussianDiffusion
from torchvision.utils import save_image
from utils import get_named_beta_schedule
from embedding import ConditionalEmbedding
from Scheduler import GradualWarmupScheduler
from dataloader_realblur import load_data, transback
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, init_process_group, destroy_process_group, all_gather, get_world_size
from torch.nn import DataParallel
def train(params:argparse.Namespace):
    device = torch.device("cuda:0")
    # load data
    dataloader = load_data(params.batchsize, params.numworkers,params.data_path)
    # initialize models
    net = Unet(
                in_ch = params.inch,
                mod_ch = params.modch,
                out_ch = params.outch,
                ch_mul = params.chmul,
                num_res_blocks = params.numres,
                cdim = 64,#params.cdim,
                use_conv = params.useconv,
                droprate = params.droprate,
                dtype = params.dtype
            )
    cemblayer = ConditionalEmbedding(10,32,64).to(device)
    # cemblayer = ConditionalEmbedding(10, params.cdim, params.cdim).to(device)

    # load last epoch
    lastpath = os.path.join(params.moddir,'last_epoch.pt')
    if os.path.exists(lastpath):
        lastepc = torch.load(lastpath)['last_epoch']
        # load checkpoints
        checkpoint = torch.load(os.path.join(params.moddir, f'ckpt_{lastepc}_checkpoint.pt'), map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        cemblayer.load_state_dict(checkpoint['cemblayer'])
    else:
        lastepc = 0
    betas = get_named_beta_schedule(num_diffusion_timesteps = params.T)
    diffusion = GaussianDiffusion(
                    dtype = params.dtype,
                    model = net,
                    betas = betas,
                    w = params.w,
                    v = params.v,
                    device = device
                )
    
    # DDP settings 
    diffusion.model = DataParallel(
                            diffusion.model,
                            device_ids = list(range(torch.cuda.device_count()))
                        )
    cemblayer = DataParallel(
                    cemblayer,
                    device_ids = list(range(torch.cuda.device_count()))
                )
    # optimizer settings
    optimizer = torch.optim.AdamW(
                    itertools.chain(
                        diffusion.model.parameters(),
                        cemblayer.parameters()
                    ),
                    lr = params.lr,
                    weight_decay = 1e-4
                )
    
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer = optimizer,
                            T_max = params.epoch,
                            eta_min = 0,
                            last_epoch = -1
                        )
    warmUpScheduler = GradualWarmupScheduler(
                            optimizer = optimizer,
                            multiplier = params.multiplier,
                            warm_epoch = params.epoch // 10,
                            after_scheduler = cosineScheduler,
                            last_epoch = lastepc
                        )
    if lastepc != 0:
        optimizer.load_state_dict(checkpoint['optimizer'])
        warmUpScheduler.load_state_dict(checkpoint['scheduler'])
    # training
    cnt = torch.cuda.device_count()
    for epc in range(lastepc, params.epoch):
        # turn into train mode
        diffusion.model.train()
        cemblayer.train()
        # sampler.set_epoch(epc)
        # batch iterations
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for img, gt in tqdmDataLoader:
                b = img.shape[0]
                optimizer.zero_grad()
                x_0 = img.to(device)
                gt = gt.to(device)
                t = torch.fft.fft2(x_0).type(torch.FloatTensor).to(device)
                # print(t.shape)
                cemb = cemblayer(t)
                cemb = torch.fft.ifft2(cemb).type(torch.FloatTensor)#.to(device)
                cemb[np.where(np.random.rand(b)<params.threshold)] = 0
                # print(cemb.shape)
                loss = diffusion.trainloss(gt, cemb = cemb)
                loss.backward()
                optimizer.step()
                tqdmDataLoader.set_postfix(
                    ordered_dict={
                        "epoch": epc + 1,
                        "loss: ": loss.item(),
                        "batch per device: ":x_0.shape[0]/torch.cuda.device_count(),
                        "img shape: ": x_0.shape[1:],
                        "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                    }
                )
        warmUpScheduler.step()
        # evaluation and save checkpoint
        if (epc + 1) % params.interval == 0:
            diffusion.model.eval()
            cemblayer.eval()
            # generating samples
            # The model generate 80 pictures(8 per row) each time
            # pictures of same row belong to the same class
            all_samples = []
            each_device_batch = params.genbatch // cnt
            with torch.no_grad():
                lab = torch.ones(params.clsnum, each_device_batch // params.clsnum).type(torch.long) \
                * torch.arange(start = 0, end = params.clsnum).reshape(-1, 1)
                lab = lab.reshape(-1, 1).squeeze()
                lab = lab.to(device)
                cemb = cemblayer(lab)
                genshape = (each_device_batch , 3, 32, 32)
                if params.ddim:
                    generated = diffusion.ddim_sample(genshape, params.num_steps, params.eta, params.select, cemb = cemb)
                else:
                    generated = diffusion.sample(genshape, cemb = cemb)
                img = transback(generated)
                img = img.reshape(params.clsnum, each_device_batch // params.clsnum, 3, 32, 32).contiguous()
                gathered_samples = [torch.zeros_like(img) for _ in range(get_world_size())]
                all_gather(gathered_samples, img)
                all_samples.extend([img for img in gathered_samples])
                samples = torch.concat(all_samples, dim = 1).reshape(params.genbatch, 3, 32, 32)
                # if local_rank == 0:
                #     save_image(samples, os.path.join(params.samdir, f'generated_{epc+1}_pict.png'), nrow = params.genbatch // params.clsnum)
            # save checkpoints
            checkpoint = {
                                'net':diffusion.model.module.state_dict(),
                                'cemblayer':cemblayer.module.state_dict(),
                                'optimizer':optimizer.state_dict(),
                                'scheduler':warmUpScheduler.state_dict()
                            }
            torch.save({'last_epoch':epc+1}, os.path.join(params.moddir,'last_epoch.pt'))
            torch.save(checkpoint, os.path.join(params.moddir, f'ckpt_{epc+1}_checkpoint.pt'))
        torch.cuda.empty_cache()
    destroy_process_group()

def main():
    # several hyperparameters for model
    parser = argparse.ArgumentParser(description='test for diffusion model')

    parser.add_argument('--batchsize',type=int,default=256,help='batch size per device for training Unet model')
    parser.add_argument('--numworkers',type=int,default=4,help='num workers for training Unet model')
    parser.add_argument('--inch',type=int,default=3,help='input channels for Unet model')
    parser.add_argument('--modch',type=int,default=64,help='model channels for Unet model')
    parser.add_argument('--T',type=int,default=1000,help='timesteps for Unet model')
    parser.add_argument('--outch',type=int,default=3,help='output channels for Unet model')
    parser.add_argument('--chmul',type=list,default=[1,2,2,2],help='architecture parameters training Unet model')
    parser.add_argument('--numres',type=int,default=2,help='number of resblocks for each block in Unet model')
    parser.add_argument('--cdim',type=int,default=10,help='dimension of conditional embedding')
    parser.add_argument('--useconv',type=bool,default=True,help='whether use convlution in downsample')
    parser.add_argument('--droprate',type=float,default=0.1,help='dropout rate for model')
    parser.add_argument('--dtype',default=torch.float32)
    parser.add_argument('--lr',type=float,default=2e-4,help='learning rate')
    parser.add_argument('--w',type=float,default=1.8,help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--v',type=float,default=0.3,help='hyperparameters for the variance of posterior distribution')
    parser.add_argument('--epoch',type=int,default=1500,help='epochs for training')
    parser.add_argument('--multiplier',type=float,default=2.5,help='multiplier for warmup')
    parser.add_argument('--threshold',type=float,default=0.1,help='threshold for classifier-free guidance')
    parser.add_argument('--interval',type=int,default=20,help='epoch interval between two evaluations')
    parser.add_argument('--moddir',type=str,default='model',help='model addresses')
    parser.add_argument('--samdir',type=str,default='sample',help='sample addresses')
    parser.add_argument('--genbatch',type=int,default=80,help='batch size for sampling process')
    parser.add_argument('--clsnum',type=int,default=10,help='num of label classes')
    parser.add_argument('--num_steps',type=int,default=50,help='sampling steps for DDIM')
    parser.add_argument('--eta',type=float,default=0,help='eta for variance during DDIM sampling process')
    parser.add_argument('--select',type=str,default='linear',help='selection stragies for DDIM')
    parser.add_argument('--ddim',type=lambda x:(str(x).lower() in ['true','1', 'yes']),default=False,help='whether to use ddim')
    parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')
    parser.add_argument('--data_path',default='RealBlur-Tele/RealBlur-J-Tele_ECC_IMCORR_centroid_itensity_ref',type=str,help='Path to blur data')

    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
