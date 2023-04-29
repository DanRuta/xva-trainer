import torch
from itertools import chain

def make_optim(args, model):

    params = [
        model.emb_l.parameters(),

        model.text_encoder.parameters(),
        model.duration_predictor.parameters(),
        model.flow.parameters(),
    ]
    if not args.frozen_vocoder_langs:
        params.append(model.posterior_encoder.parameters())
        params.append(model.waveform_decoder.parameters())

    if args.mltts_rc:
        params.append(model.reversal_classifier.parameters())

    if args.pitch:
        print("== Adding pitch params to optim")
        params.append(model.pitch_predictor.parameters())
        if not args.ow_flow:
            params.append(model.pitch_emb.parameters())
    if args.energy:
        print("== Adding energy params to optim")
        params.append(model.energy_predictor.parameters())
        if not args.ow_flow:
            params.append(model.energy_emb.parameters())

    if args.hifi_only:
        params = [model.posterior_encoder.parameters(), model.waveform_decoder.parameters()]

    gen_parameters = chain(*params)


    if args.bnb:
        import bitsandbytes as bnb
        print("============= Using bitsandbytes optim =============")
        # optimizer0 = bnb.optim.AdamW(gen_parameters, lr=args.lr, betas=[0.8, 0.99], eps=1e-09, weight_decay=0.01)
        optimizer0 = bnb.optim.Adam8bit(gen_parameters, lr=args.lr, betas=[0.8, 0.99], eps=1e-09, weight_decay=0.01)
        # optimizer1 = bnb.optim.AdamW(model.disc.parameters(), lr=0.0002, betas=[0.8, 0.99], eps=1e-09, weight_decay=0.01)
        optimizer1 = bnb.optim.Adam8bit(model.disc.parameters(), lr=0.0002, betas=[0.8, 0.99], eps=1e-09, weight_decay=0.01)
    else:
        if args.lion:
            from lion_pytorch import Lion
            print("============= Using Lion optim =============")
            # gpus=1,2 bs=30 target=400 | 22.3GB, ~18k
            # gpus=1,2 bs=30 target=400 | 22.1GB, ~18k
            optimizer0 = Lion(gen_parameters, lr=0.0002/5, betas=[0.8, 0.99], weight_decay=0.01*5)
            optimizer1 = Lion(model.disc.parameters(), lr=0.0002/5, betas=[0.8, 0.99], weight_decay=0.01*5)
        else:
            # gpus=1,2 bs=30 target=400 | 23.8GB, ~17k
            # gpus=1,2 bs=30 target=400 | 20.8GB, ~17.5k
            # gpus=1,2 bs=30 target=400 | 22.9GB, ~18k
            optimizer0 = torch.optim.AdamW(gen_parameters, lr=args.lr, betas=[0.8, 0.99], eps=1e-09, weight_decay=0.01)
            optimizer1 = torch.optim.AdamW(model.disc.parameters(), lr=0.0002, betas=[0.8, 0.99], eps=1e-09, weight_decay=0.01)


    return [optimizer0, optimizer1]

def get_scheduler(args, model, optimizer, restore_step):

    scheduler0 = torch.optim.lr_scheduler.ExponentialLR(optimizer[0], gamma=0.999875, last_epoch=-1)
    scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer[1], gamma=0.999875, last_epoch=-1)

    if args.continue_path:
        scheduler0.last_epoch = restore_step
        scheduler1.last_epoch = restore_step

    return [scheduler0, scheduler1]

def format_time (seconds):
    time_str = ""
    if seconds>60*60*24:
        days = int(seconds/(60*60*24))
        time_str += f'{days}d '
        seconds -= days*(60*60*24)
    if seconds>60*60:
        hours = int(seconds/(60*60))
        time_str += f'{hours}h '
        seconds -= hours*(60*60)
    if seconds>60:
        minutes = int(seconds/(60))
        time_str += f'{minutes}m '
        seconds -= minutes*(60)
    if seconds>0:
        time_str += f'{int(seconds)}s '

    return time_str