import os
import json
import traceback

# Not a model, but it was easier to just integrate the code this way

import multiprocessing as mp
from lib.ffmpeg_normalize._ffmpeg_normalize import FFmpegNormalize


def normalizeTask (data):
    [inPath, outPath] = data

    sample_rate = 22050
    ffmpeg_normalize = FFmpegNormalize(
            normalization_type="ebu",
            target_level=-23.0,
            print_stats=False,
            loudness_range_target=7.0,
            true_peak=-2.0,
            offset=0.0,
            dual_mono=False,
            audio_codec=None,
            audio_bitrate=None,
            sample_rate=sample_rate,
            keep_original_audio=False,
            pre_filter=None,
            post_filter=None,
            video_codec="copy",
            video_disable=False,
            subtitle_disable=False,
            metadata_disable=False,
            chapters_disable=False,
            extra_input_options=[],
            extra_output_options=[],
            output_format=None,
            dry_run=False,
            progress=False,
        )

    try:
        ffmpeg_normalize.add_media_file(inPath, outPath)
        ffmpeg_normalize.run_normalization()
    except:
        print(traceback.format_exc())
        return "stderr: "+ traceback.format_exc()

    return None

class AudioNormalizer(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(AudioNormalizer, self).__init__()

        self.logger = logger
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = device
        self.ckpt_path = None

        self.model = None
        self.isReady = True


    def load_state_dict (self, ckpt_path, sd):
        pass

    def set_device (self, device):
        pass

    def runTask (self, data, websocket=None):
        return self.normalize(data, websocket)


    async def normalize(self, data, websocket):

        inPath, outputDirectory = data["inPath"], data["outputDirectory"]
        useMP = data["toolSettings"]["useMP"]
        # processes = data["toolSettings"]["mpProcesses"]
        processes = max(1, mp.cpu_count()-1) # TODO

        if useMP:

            input_paths = sorted(os.listdir(inPath))
            output_paths = [f'{outputDirectory}/{fpath}' for fpath in input_paths]
            input_paths = [f'{inPath}/{fpath}' for fpath in input_paths]


            workItems = []
            for ip, path in enumerate(input_paths):
                workItems.append([path, output_paths[ip]])

            workers = processes if processes>0 else max(1, mp.cpu_count()-1)
            workers = min(len(workItems), workers)

            self.logger.info("[mp ffmpeg] workers: "+str(workers))

            pool = mp.Pool(workers)
            results = pool.map(normalizeTask, workItems)
            pool.close()
            pool.join()
            self.logger.info("Finished mp job")

            errs = [items for items in results if items is not None]
            if len(errs):
                self.logger.info(errs)
                if websocket is not None:
                    await websocket.send(json.dumps({"key": "tasks_error", "data": f'Task done. {len(errs)} items failed (out of: {len(input_paths)})<br>First error (check the server.log for all):<br>{errs[0]}'}))

        else:

            outputPath = f'{outputDirectory}/{inPath.split("/")[-1].split(".")[0]}.wav'

            sample_rate = 22050
            ffmpeg_normalize = FFmpegNormalize(
                    normalization_type="ebu",
                    target_level=-23.0,
                    print_stats=False,
                    loudness_range_target=7.0,
                    true_peak=-2.0,
                    offset=0.0,
                    dual_mono=False,
                    audio_codec=None,
                    audio_bitrate=None,
                    sample_rate=sample_rate,
                    keep_original_audio=False,
                    pre_filter=None,
                    post_filter=None,
                    video_codec="copy",
                    video_disable=False,
                    subtitle_disable=False,
                    metadata_disable=False,
                    chapters_disable=False,
                    extra_input_options=[],
                    extra_output_options=[],
                    output_format=None,
                    dry_run=False,
                    progress=False,
                )

            try:
                ffmpeg_normalize.add_media_file(inPath, outputPath)
                ffmpeg_normalize.run_normalization()
            except:
                self.logger.info(traceback.format_exc())

            if websocket is not None:
                await websocket.send(json.dumps({"key": "tasks_next"}))

