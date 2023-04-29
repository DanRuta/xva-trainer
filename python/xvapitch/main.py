import os
import argparse
import traceback
import warnings
warnings.simplefilter(action='ignore', category=Warning)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

import torch

from xva_train import xVAPitchTrainer, sort_xvap



def init_training_run(data_folder, ckpt_fname, gpus):

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True

    trainer = xVAPitchTrainer(None, False, gpus, None, websocket=None, amp=True, cmd_training=True)

    dataset_output = f'D:/OUTPUT/{data_folder}'
    trainer.init_logs(dataset_output=dataset_output)


    # Get checkpoint
    final_ckpt_fname = None
    if os.path.exists(f'{dataset_output}'): # Check for existing checkpoints
        ckpts = os.listdir(f'{dataset_output}')
        ckpts = [ckpt for ckpt in ckpts if ckpt.startswith("xVAPitch_")]
        if len(ckpts):
            ckpts = sorted(ckpts, key=sort_xvap)
            final_ckpt_fname = f'{dataset_output}/{ckpts[-1]}'
        if final_ckpt_fname is None:
            if ckpt_fname=="[base]":
                final_ckpt_fname = trainer.pretrained_ckpt
            else:
                ckpt_is_dir = os.path.isdir(ckpt_fname)
                if final_ckpt_fname is None and ckpt_is_dir:
                    ckpts = os.listdir(f'{ckpt_fname}')
                    ckpts = [ckpt for ckpt in ckpts if ckpt.startswith("xVAPitch_")]

                    if len(ckpts):
                        ckpts = sorted(ckpts, key=sort_xvap)
                        final_ckpt_fname = f'{ckpt_fname}/{ckpts[-1]}'

                if final_ckpt_fname is None:
                    final_ckpt_fname = ckpt_fname


    return trainer, final_ckpt_fname


async def do_voice(training_queue, trainer, force_stage, dataset_path, final_ckpt_fname, bs, workers, err_counter):

    data = {}
    if force_stage:
        data["force_stage"] = force_stage
    data["dataset_path"] = dataset_path
    # data["output_path"] = f'D:/OUTPUT/{dataset_path.split("/")[-1]}'
    data["output_path"] = f'D:/OUTPUT/'
    data["checkpoint"] = final_ckpt_fname
    data["num_workers"] = workers
    data["batch_size"] = bs
    data["use_amp"] = "true"

    try:
        await trainer.start(data, gpus, resume=False)
    except KeyboardInterrupt:
        trainer.running = False
        raise
    except RuntimeError as e:
        trainer.running = False
        try:
            del trainer.train_loader
        except:
            pass
        try:
            del trainer.dataloader_iterator
            del trainer.model
            del trainer.criterion
            del trainer.trainset
            del trainer.optimizer
        except:
            pass

        torch.cuda.empty_cache()

        if "out of memory" in str(e):
            print(f'CUDA OOM, Reducing batch size...')
            if trainer:
                trainer.print_and_log(trainer.training_log, f'============= Reducing batch size from {bs} to {bs-5}')
            bs -= 5

        # elif trainer and trainer.END_OF_TRAINING:
        #     if trainer:
        #         trainer.print_and_log(trainer.training_log, "=====Moving on...")
        #     del training_queue[0]
        #     bs = args.batch_size
        #     err_counter = 0
        elif not trainer or (trainer and not trainer.END_OF_TRAINING):
            if trainer:
                trainer.print_and_log(trainer.training_log, traceback.format_exc())
            err_counter += 1

        if err_counter>10 or trainer.END_OF_TRAINING:
            if trainer:
                trainer.print_and_log(trainer.training_log, "=====Moving on...")
            del training_queue[0]
            bs = args.batch_size
            err_counter = 0


        do_next_voice(training_queue, bs, err_counter)




if __name__ == '__main__':

    # python main.py -gpus=0 -bs=24

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpus', default="0")
    parser.add_argument('-ckpt', default="[base]")
    parser.add_argument("-bs", '--batch_size', type=int, default=32)
    parser.add_argument("-w", '--workers', type=int, default=4)
    args, unk_args = parser.parse_known_args()
    bs = args.batch_size
    gpus = [int(val) for val in args.gpus.split(",")]


    training_queue = []

    if args.gpus=="0":
        training_queue.append([f'D:/xVASpeech/GAME_DATA/de_sk_femaleeventoned_F'])


    async def do_next_voice(training_queue, bs, err_counter):
        data_folder = training_queue[0][0].split("/")[-1]
        force_stage = training_queue[0][1] if len(training_queue[0])==2 else None
        print(f'\n\n================\nNEXT VOICE: {data_folder}\n================\n\n')

        trainer = None

        trainer, final_ckpt_fname = init_training_run(data_folder, args.ckpt, gpus)
        await do_voice(training_queue, trainer, force_stage, training_queue[0][0], final_ckpt_fname, bs, args.workers, err_counter)


    # Create a python event loop for the async function
    # Thanks ChatGPT
    async def main():
        # Start the event loop
        loop = asyncio.get_event_loop()

        # Schedule the async function to run
        task = loop.create_task(do_next_voice(training_queue, args.batch_size, 0))

        # Wait for the async function to complete
        await task

    import asyncio
    asyncio.run(main())
