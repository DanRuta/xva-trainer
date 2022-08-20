import os
import json
import traceback

import subprocess

# Not a model, but it was easier to just integrate the code this way

import multiprocessing as mp


def splitTask (data):
    [ffmpeg_path, inPath, t1, t2, outname_temp, outname, PROD] = data

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    # Add some padding
    # t1 = time_minus(t1, "00:00:01")
    # t2 = time_plus(t2, "00:00:01")

    t1_delta = time_minus(t1, "00:00:01")
    t2_delta = time_diff(t1, t2)

    command = f'{ffmpeg_path} -ss {t1_delta} -i {inPath} -ss 00:00:01 -t {t2_delta} -vn -acodec copy -c copy {outname_temp} -map a -ss 00:00:01 -t {t2_delta} -ar 22050 -ac 1 {outname}'
    command_process = subprocess.Popen(command, startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = command_process.communicate()
    stderr = stderr.decode("utf-8")

    with lock:
        progress.value += 1
        with open(f'{"./resources/app" if PROD else "."}/python/srt_split/.progress.txt', "w+") as f:
            f.write(f'{(int(progress.value)/total.value*100*100)/100}')


def time_minus (t1, t2):
    sec_diff = int(t2.split(":")[2])
    min_diff = int(t2.split(":")[1])
    hour_diff = int(t2.split(":")[0])

    t1_sec = int(t1.split(":")[2])
    t1_min = int(t1.split(":")[1])
    t1_hour = int(t1.split(":")[0])


    final_sec = t1_sec-sec_diff if sec_diff else t1_sec
    t1_min = t1_min-1 if final_sec<0 else t1_min
    final_sec = (60-final_sec) if final_sec<0 else final_sec

    final_min = t1_min-min_diff if min_diff else t1_min
    t1_hour = t1_hour-1 if final_min<0 else t1_hour
    final_min = (60-final_min) if final_min<0 else final_min

    final_hour = t1_hour-hour_diff if hour_diff else t1_hour
    final_hour = (24-final_hour) if final_hour<0 else final_hour

    final_stamp = f'{final_hour}:{final_min}:{final_sec}'
    return final_stamp

def time_plus (t1, t2):
    sec_diff = int(t2.split(":")[2])
    min_diff = int(t2.split(":")[1])
    hour_diff = int(t2.split(":")[0])

    t1_sec = int(t1.split(":")[2])
    t1_min = int(t1.split(":")[1])
    t1_hour = int(t1.split(":")[0])

    final_sec = t1_sec+sec_diff if sec_diff else t1_sec
    t1_min = t1_min+1 if final_sec>60 else t1_min
    final_sec = (final_sec-60) if final_sec>60 else final_sec

    final_min = t1_min+min_diff if min_diff else t1_min
    t1_hour = t1_hour+1 if final_min>60 else t1_hour
    final_min = (final_min-60) if final_min>60 else final_min

    final_hour = t1_hour+hour_diff if hour_diff else t1_hour
    final_hour = (final_hour-24) if final_hour>24 else final_hour

    final_stamp = f'{final_hour}:{final_min}:{final_sec}'
    return final_stamp

def time_diff (t1, t2):
    sec_diff = int(t2.split(":")[2])-int(t1.split(":")[2])
    min_diff = int(t2.split(":")[1])-int(t1.split(":")[1])
    hour_diff = int(t2.split(":")[0])-int(t1.split(":")[0])

    final_sec = (60-sec_diff) if sec_diff<0 else sec_diff
    min_diff = (min_diff-1) if sec_diff<0 else min_diff
    final_min = (60-min_diff) if min_diff<0 else min_diff
    hour_diff = (hour_diff-1) if min_diff<0 else hour_diff
    final_hour = (24-hour_diff) if hour_diff<0 else hour_diff

    final_stamp = f'{str(final_hour).zfill(2)}:{str(final_min).zfill(2)}:{str(final_sec).zfill(2)}'
    return final_stamp


def initializer(*arguments):
    global progress, total, lock
    progress, total, lock = arguments

class SRTSplit(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(SRTSplit, self).__init__()

        self.logger = logger
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = device
        self.ckpt_path = None

        self.model = None
        self.isReady = True
        self.ffmpeg_path = f'{"./resources/app" if self.PROD else "."}/python/ffmpeg.exe'


    def load_state_dict (self, ckpt_path, sd):
        pass

    def set_device (self, device):
        pass

    def runTask (self, data, websocket=None):
        return self.srt_split(data, websocket)


    def split_sync(self, data):

        [ffmpeg_path, inPath, t1, t2, outname_temp, outname, PROD] = data

        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        # Add some padding
        # t1 = time_minus(t1, "00:00:01")
        # t2 = time_plus(t2, "00:00:01")

        if t1=="00:00:00":
            t1_delta = t1
        else:
            t1_delta = time_minus(t1, "00:00:01")
        t2_delta = time_diff(t1, t2)

        command = f'{ffmpeg_path} -ss {t1_delta} -i "{inPath}" -ss 00:00:01 -t {t2_delta} -vn -acodec copy -c copy "{outname_temp}" -map a -ss 00:00:01 -t {t2_delta} -ar 22050 -ac 1 "{outname}"'
        command_process = subprocess.Popen(command, startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = command_process.communicate()
        stderr = stderr.decode("utf-8")


    async def srt_split(self, data, websocket):

        inPath, outputDirectory = data["inPath"], data["outputDirectory"]
        useMP = data["toolSettings"]["useMP"] if "useMP" in data["toolSettings"].keys() else False
        processes = max(1, int(mp.cpu_count()/2)-5) # TODO, figure out why more processes break the websocket

        files = [fname for fname in sorted(os.listdir(inPath)) if fname.endswith(".srt")]

        try:
            os.remove(f'{"./resources/app" if self.PROD else "."}/python/srt_split/.progress.txt')
        except:
            pass

        metadata = []

        os.makedirs(f'{outputDirectory}/wavs/', exist_ok=True)

        data_splits = []
        for srt_file in files:

            video_file_basename = [fname for fname in sorted(os.listdir(inPath)) if not fname.endswith(".srt") and ".".join(srt_file.split(".")[:-1]) in ".".join(fname.split(".")[:-1])][0]
            video_file_name = f'{inPath}/{video_file_basename}'

            with open(f'{inPath}/{srt_file}') as f:
                lines = f.read().split("\n")

                current_timing = None
                current_transcript = []

                for line in lines:
                    if not len(line.strip()) and current_timing and len(current_transcript):
                        prefix = str(len(data_splits)).zfill(7)
                        inPath_base = ".".join(video_file_basename.split(".")[:-1])
                        outname_temp = f'{outputDirectory}/wavs/{prefix}_{inPath_base}.aac'
                        outname = f'{outputDirectory}/wavs/{prefix}_{inPath_base}.wav'

                        metadata.append(f'{prefix}_{inPath_base}.wav|{",".join(current_transcript)}|{",".join(current_transcript)}')

                        data_splits.append([self.ffmpeg_path, video_file_name, current_timing[0], current_timing[1], outname_temp, outname, self.PROD])
                        current_timing = None
                        current_transcript = []

                    elif "-->" in line:
                        current_timing = [line.split("-->")[0].split(",")[0], line.split("-->")[1].split(",")[0]]

                    elif current_timing is not None:
                        current_transcript.append(line)

        if useMP:
            workers = processes if processes>0 else max(1, mp.cpu_count()-1)
            workers = min(len(data_splits), workers)

            progress = mp.Value("i", 0)
            total = mp.Value("i", 0)
            lock = mp.Lock()
            total.value = len(data_splits)

            pool = mp.Pool(workers, initializer, (progress, total, lock))
            _ = pool.map(splitTask, data_splits)
            pool.close()
            pool.join()

            with open(f'{outputDirectory}/metadata.csv', "w+") as f:
                f.write("\n".join(metadata))


            files = os.listdir(f'{outputDirectory}/wavs/')
            for file in files:
                if ".wav" not in file and ".csv" not in file:
                    os.remove(f'{outputDirectory}/wavs/{file}')
            try:
                os.remove(f'{"./resources/app" if self.PROD else "."}/python/srt_split/.progress.txt')
            except:
                pass


            if websocket is not None:
                await websocket.send(json.dumps({"key": "tasks_next"}))

        else:
            for dsi, data_split in enumerate(data_splits):
                self.split_sync(data_split)

                if dsi%3==0 and websocket is not None:
                    with open(f'{"./resources/app" if self.PROD else "."}/python/srt_split/.progress.txt', "w+") as f:
                        f.write(f'{(int(dsi+1)/len(data_splits)*100*100)/100}%')


            with open(f'{outputDirectory}/metadata.csv', "w+") as f:
                f.write("\n".join(metadata))


            files = os.listdir(f'{outputDirectory}/wavs/')
            for file in files:
                if ".wav" not in file and ".csv" not in file:
                    os.remove(f'{outputDirectory}/wavs/{file}')
            try:
                os.remove(f'{"./resources/app" if self.PROD else "."}/python/srt_split/.progress.txt')
            except:
                pass


            if websocket is not None:
                await websocket.send(json.dumps({"key": "tasks_next"}))

