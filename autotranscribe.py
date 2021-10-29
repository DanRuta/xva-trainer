import os
import sys
import shutil
import traceback
import warnings

import tqdm
import torch
warnings.simplefilter(action='ignore', category=Warning)

import soundfile as sf
from transformers import Wav2Vec2ForMaskedLM, Wav2Vec2Tokenizer

# load pretrained model
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForMaskedLM.from_pretrained("facebook/wav2vec2-base-960h")

print("sys.argv", sys.argv)
# metadata_path = f'./out/partial_meta.csv'
metadata_path = sys.argv[1]

with open(metadata_path, "r") as f:
    lines = f.read().split("\n")

out_lines = []
for line in lines:
    text = line.split("|")[1]
    if len(text):
        out_lines.append(line)
    else:
        try:
            # load audio
            audio_file_name = "/".join(metadata_path.split("/")[:-1])+"/wavs/"+line.split("|")[0].replace(".wav","")+".wav"
            print("audio_file_name", audio_file_name)
            audio_input, _ = sf.read(audio_file_name)

            input_values = tokenizer(audio_input, return_tensors="pt").input_values
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = tokenizer.batch_decode(predicted_ids)[0]
            print(f'transcription, {transcription}')

            out_lines.append(f'{line.split("|")[0]}|{transcription.lower()}|{transcription.lower()}')
        except KeyboardInterrupt :
            raise
        except:
            print(traceback.format_exc())
            out_lines.append(line)

with open(metadata_path, "w") as f:
    f.write("\n".join(out_lines))