import os
import sys
import gc
import traceback
import multiprocessing
import wave
import contextlib
import numpy as np

if __name__ == '__main__':
    multiprocessing.freeze_support()

    PROD = False
    # PROD = True
    CPU_ONLY = False
    # CPU_ONLY = True

    # Saves me having to do backend re-compilations for every little UI hotfix
    with open(f'{"./resources/app" if PROD else "."}/javascript/script.js', encoding="utf8") as f:
        lines = f.read().split("\n")
        APP_VERSION = lines[1].split('"')[1]

    # configurable in ports.txt
    SERVER_PORT = 8002
    WEBSOCKET_PORT = 8001




    # Imports and logger setup
    # ========================
    try:
        import asyncio
        import websockets
        import _thread
        import python.pyinstaller_imports
        import numpy
        from python.audio_norm.model import AudioNormalizer

        import logging
        from logging.handlers import RotatingFileHandler
        import json
        from http.server import BaseHTTPRequestHandler, HTTPServer
    except:
        print(traceback.format_exc())
        with open("./DEBUG_err_imports.txt", "w+") as f:
            f.write(traceback.format_exc())

    # Pyinstaller hack
    # ================
    try:
        def script_method(fn, _rcb=None):
            return fn
        def script(obj, optimize=True, _frames_up=0, _rcb=None):
            return obj
        import torch.jit
        torch.jit.script_method = script_method
        torch.jit.script = script
        import torch
    except:
        with open("./DEBUG_err_import_torch.txt", "w+") as f:
            f.write(traceback.format_exc())
    # ================

    try:
        logger = logging.getLogger('serverLog')
        logger.setLevel(logging.DEBUG)
        # fh = RotatingFileHandler('{}/server.log'.format(os.path.dirname(os.path.realpath(__file__))), maxBytes=5*1024*1024, backupCount=2)
        fh = RotatingFileHandler('./server.log', maxBytes=2*1024*1024, backupCount=5)
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.info(f'New session. Version: {APP_VERSION}. Installation: {"CPU" if CPU_ONLY else "CPU+GPU"}')

        logger.orig_info = logger.info

        def prefixed_log (msg):
            logger.info(f'{logger.logging_prefix}{msg}')


        def set_logger_prefix (prefix=""):
            if len(prefix):
                logger.logging_prefix = f'[{prefix}]: '
                logger.log = prefixed_log
            else:
                logger.log = logger.orig_info

        logger.set_logger_prefix = set_logger_prefix
        logger.set_logger_prefix("")

    except:
        with open("./DEBUG_err_logger.txt", "w+") as f:
            f.write(traceback.format_exc())
        try:
            logger.info(traceback.format_exc())
        except:
            pass
    # ========================




    # ======================== Models manager
    try:
        from python.models_manager import ModelsManager
        models_manager = ModelsManager(logger, PROD, device="cpu")
    except:
        logger.info("Models manager failed to initialize")
        logger.info(traceback.format_exc())
    # ========================



    print("Models ready")
    logger.info("Models ready")

    try:
        with open(f'{"./resources/app" if PROD else "."}/ports.txt') as f:
            lines = f.read().split("\n")
            SERVER_PORT = int(lines[0].split(",")[1].strip())
            WEBSOCKET_PORT = int(lines[1].split(",")[1].strip())
    except:
        logger.info(traceback.format_exc())
        pass


    async def websocket_handler(websocket, path):
        async for message in websocket:
            try:
                message = json.loads(message)
                model = message["model"]
                gpus = [int(g) for g in message["gpus"].split(",")] if "gpus" in message else [0]
                task = message["task"] if "task" in message else None
                data = message["data"] if "data" in message else None


                # DEBUG
                # ==================
                if model=="exit":
                    sys.exit()
                if model=="print":
                    logger.info(data)
                    await websocket.send("")
                if model=="print_and_return":
                    logger.info(data)
                    await websocket.send(data)
                if model=="getTimedData":
                    import time
                    await websocket.send("1")
                    time.sleep(1)
                    await websocket.send("2")
                    time.sleep(1)
                    await websocket.send("3")
                # ==================


                # Training
                if task in ["startTraining", "resume", "pause", "stop"]:
                    try:
                        if task=="startTraining" or task=="resume":
                            # _thread.start_new_thread(between_callback, (models_manager, data, websocket, [0], task=="resume"))
                            _thread.start_new_thread(between_callback, (models_manager, data, websocket, gpus, task=="resume"))
                        else:
                            if task=="pause":
                                logger.info("server.py pause")
                                if "fastpitch1_1" not in models_manager.models_bank.keys() or models_manager.models_bank["fastpitch1_1"]=="move to hifi":
                                    if "hifigan" in models_manager.models_bank.keys():
                                        models_manager.models_bank["hifigan"].pause()
                                else:
                                    models_manager.models_bank["fastpitch1_1"].pause()
                            if task=="stop":
                                if "fastpitch1_1" in models_manager.models_bank.keys():
                                    del models_manager.models_bank["fastpitch1_1"]
                                if "hifigan" in models_manager.models_bank.keys():
                                    del models_manager.models_bank["hifigan"]

                                gc.collect()
                                torch.cuda.empty_cache()
                    except KeyboardInterrupt:
                        sys.exit()
                    except:
                        logger.info(f'TRAINING_ERROR:{traceback.format_exc()}')
                        await websocket.send(f'TRAINING_ERROR:{traceback.format_exc()}')
                else:
                    # Tasks
                    await models_manager.init_model(model, websocket)
                    if task=="runTask":
                        logger.info(f'Task: {model}')
                        try:
                            await models_manager.models_bank[model].runTask(data, websocket=websocket)
                        except:
                            logger.info(traceback.format_exc())
                            await websocket.send(f'ERROR:{traceback.format_exc()}')

            except KeyboardInterrupt:
                sys.exit()
            except:
                logger.info(f'message: {message} | {traceback.format_exc()}')

    # https://stackoverflow.com/questions/59645272/how-do-i-pass-an-async-function-to-a-thread-target-in-python
    def between_callback(models_manager, data, websocket, gpus, resume):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(handleTrainingLoop(models_manager, data, websocket, gpus, resume))
        loop.close()

    async def handleTrainingLoop(models_manager, data, websocket, gpus, resume):
        try:
            if ("fastpitch1_1" in models_manager.models_bank.keys() and models_manager.models_bank["fastpitch1_1"] == "move to hifi") or (data is not None and "force_stage" in data.keys() and data["force_stage"]==5) or ("fastpitch1_1" not in models_manager.models_bank.keys() and "hifigan" in models_manager.models_bank.keys()):
                from python.hifigan.xva_train import handleTrainer as handleTrainer_hifi
                result = await handleTrainer_hifi(models_manager, data, websocket, gpus=gpus, resume=resume)
                if result == "done":
                    logger.info("server.py done training hifigan")
            else:
                if not ("hifigan" in models_manager.models_bank.keys() and resume):
                    from python.fastpitch1_1.xva_train import handleTrainer as handleTrainer_fp
                    result = await handleTrainer_fp(models_manager, data, websocket, gpus=gpus, resume=resume)

                if result == "move to hifi":
                    logger.info("server.py moving on to HiFi training")
                    return await handleTrainingLoop(models_manager, data, websocket, gpus, False)
        except:
            logger.info(f'TRAINING_ERROR:{traceback.format_exc()}')
            await websocket.send(f'TRAINING_ERROR:{traceback.format_exc()}')


    def get_or_create_eventloop ():
        try:
            return asyncio.get_event_loop()
        except RuntimeError as ex:
            if "There is no current event loop in thread" in str(ex):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return asyncio.get_event_loop()
            else:
                logger.info(str(ex))
    def startWebSocket ():
        try:
            logger.info("Starting websocket")
            get_or_create_eventloop()
            start_server = websockets.serve(websocket_handler, "localhost", WEBSOCKET_PORT)
            loop = asyncio.get_event_loop()
            loop.run_until_complete(start_server)
            loop.run_forever()

            startWebSocket()
        except:
            import traceback
            with open("DEBUG_websocket.txt", "w+") as f:
                print(traceback.format_exc())
                logger.info(traceback.format_exc())
                f.write(traceback.format_exc())



    # Server
    class Handler(BaseHTTPRequestHandler):
        def _set_response(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()

        def do_GET(self):
            returnString = "[DEBUG] Get request for {}".format(self.path).encode("utf-8")
            logger.info(returnString)
            self._set_response()
            self.wfile.write(returnString)

        def do_POST(self):
            post_data = ""
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = json.loads(self.rfile.read(content_length).decode('utf-8'))
                req_response = "POST request for {}".format(self.path)

                if self.path == "/stopServer":
                    logger.info("POST {}".format(self.path))
                    logger.info("STOPPING SERVER")
                    sys.exit()

                if self.path == "/setDevice":
                    logger.info("POST {}".format(self.path))
                    logger.info(post_data)

                    clearTheCache = False
                    if not CPU_ONLY and models_manager.device_label=="gpu" and post_data["device"]=="gpu":
                        clearTheCache = True
                        logger.info("CLEARING CACHE")
                        torch.cuda.empty_cache()

                    use_gpu = post_data["device"]=="gpu"
                    models_manager.set_device('cuda' if use_gpu else 'cpu')

                    if clearTheCache:
                        logger.info("CLEARING CACHE")
                        torch.cuda.empty_cache()

                if self.path == "/checkReady":
                    use_gpu = post_data["device"]=="gpu"
                    models_manager.set_device('cuda' if use_gpu else 'cpu')
                    req_response = "ready"

                if self.path == "/exportWav":

                    fp_ckpt = post_data["fp_ckpt"]
                    hg_ckpt = post_data["hg_ckpt"]
                    out_path = post_data["out_path"]
                    out_path_intermediate = out_path.replace(".wav", "_temp.wav")

                    models_manager.load_model("infer_fastpitch1_1", fp_ckpt)
                    models_manager.load_model("infer_hifigan", hg_ckpt)
                    logger.info(f'Generating audio preview...')

                    req_response = models_manager.models("infer_fastpitch1_1").infer(None, "This is what my voice sounds like", out_path_intermediate, vocoder="infer_hifigan", speaker_i=None)


                    logger.info(f'Normalizing audio preview...')
                    normalizer = AudioNormalizer(logger, PROD, models_manager.device, models_manager)
                    normalizer.normalize_sync(out_path_intermediate, out_path)
                    logger.info(f'Exported.')
                    os.remove(out_path_intermediate)


                if self.path == "/getAudioLengthOfDir":

                    directory = post_data["directory"]
                    audio_lengths = []
                    files = os.listdir(directory)
                    for fname in files:
                        if not fname.endswith(".wav"):
                            continue
                        with contextlib.closing(wave.open(f'{directory}/{fname}', 'r')) as f:
                            frames = f.getnframes()
                            rate = f.getframerate()
                            duration = frames / float(rate)
                            audio_lengths.append(duration)
                    req_response = f'{np.mean(audio_lengths)}|{np.sum(audio_lengths)}'


                self._set_response()
                self.wfile.write(req_response.encode("utf-8"))
            except Exception as e:
                with open("./DEBUG_request.txt", "w+") as f:
                    f.write(traceback.format_exc())
                    f.write(str(post_data))
                logger.info("Post Error:\n {}".format(repr(e)))
                print(traceback.format_exc())
                logger.info(traceback.format_exc())

    try:
        server = HTTPServer(("",SERVER_PORT), Handler)
    except:
        with open("./DEBUG_server_error.txt", "w+") as f:
            f.write(traceback.format_exc())
        logger.info(traceback.format_exc())

    try:
        logger.info("About to start websocket")
        _thread.start_new_thread(startWebSocket, ())
        logger.info(f'Started websocket | Port: {WEBSOCKET_PORT}')

        # plugin_manager.run_plugins(plist=plugin_manager.plugins["start"]["post"], event="post start", data=None)
        print("Server ready")
        logger.info(f'Server ready | Port: {SERVER_PORT}')
        server.serve_forever()


    except KeyboardInterrupt:
        pass
    except:
        with open("./DEBUG_websocket_server_error.txt", "w+") as f:
            f.write(traceback.format_exc())
        logger.info(traceback.format_exc())
    server.server_close()