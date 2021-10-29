import os
import sys
import traceback
import json

import asyncio
import websockets

import _thread

import logging
from logging.handlers import RotatingFileHandler


import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)


APP_VERSION = "1.0"

try:
    logger = logging.getLogger('serverLog')
    logger.setLevel(logging.DEBUG)
    fh = RotatingFileHandler('{}/audio_source_separation.log'.format(os.path.dirname(os.path.realpath(__file__))), maxBytes=5*1024*1024, backupCount=2)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f'New session. Version: {APP_VERSION}')

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







async def handler(websocket, path):
    async for message in websocket:
        await websocket.send(message)

    # async for message in websocket:
        try:
            # in_data = await websocket.recv()
            logger.info(f'message: {message}')
            logger.info(f'path: {path}')

            message = json.loads(message)
            category = message["category"]
            data = message["data"] if "data" in message else None


            # await websocket.send(message)
            logger.info(f'data: {data}')

            # category, in_data = in_data.split("\n")# if "\n" in in_data else [in_data, None]


            if category=="exit":
                sys.exit()



            if category=="print":
                logger.info(data)
                await websocket.send("")



            if category=="print_and_return":
                logger.info(data)
                await websocket.send(data)


            if category=="getTimedData":
                import time
                await websocket.send("1")
                time.sleep(1)
                await websocket.send("2")
                time.sleep(1)
                await websocket.send("3")


        except KeyboardInterrupt:
            sys.exit()
        except:
            logger.info(traceback.format_exc())



def get_or_create_eventloop ():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()


def startWebSocket ():
    print("0")
    loop = get_or_create_eventloop()
    start_server = websockets.serve(handler, "localhost", 8000)
    # start_server.serve_forever()
    print("1")
    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)
    asyncio.get_event_loop().run_until_complete(start_server)
    # loop.run_until_complete(start_server)

    print("2")
    asyncio.get_event_loop().run_forever()
    # loop.run_forever()

# def startWebSocket2 ():

#     if events._get_running_loop() is not None:
#         raise RuntimeError(
#             "asyncio.run() cannot be called from a running event loop")
#     if not coroutines.iscoroutine(main):
#         raise ValueError("a coroutine was expected, got {!r}".format(main))

#     loop = events.new_event_loop()
#     try:
#         events.set_event_loop(loop)
#         loop.set_debug(debug)
#         return loop.run_until_complete(main)
#     finally:
#         try:
#             _cancel_all_tasks(loop)
#             loop.run_until_complete(loop.shutdown_asyncgens())
#         finally:
#             events.set_event_loop(None)
#             loop.close()



#     print("0")
#     start_server = websockets.serve(handler, "localhost", 8000)
#     # start_server.serve_forever()
#     print("1")
#     # loop = asyncio.new_event_loop()
#     # asyncio.set_event_loop(loop)
#     asyncio.get_or_create_eventloop().run_until_complete(start_server)
#     # loop.run_until_complete(start_server)

#     print("2")
#     asyncio.get_or_create_eventloop().run_forever()
#     loop.run_forever()


try:
    _thread.start_new_thread(startWebSocket, ())
    print("3")

    import time
    time.sleep(5)

except KeyboardInterrupt:
    sys.exit()
except:
    sys.exit()