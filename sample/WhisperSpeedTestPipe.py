import time

import whisper

from subprocess import CalledProcessError, run
import numpy as np
import torch



if __name__ == '__main__':
    # 115上,conda activate whisper
    # 167上,conda activate niutrans
    # path="/home/gaochenghao/data/whisper_model"


    time_start = time.time()
    print("start loading...")
    model = whisper.load_model("large-v2",device=torch.device('cuda:7')) # tiny large-v2
    model.eval()
    time_end = time.time()
    print("load finish：{:.2f}ms".format((time_end - time_start) * 1000))

    # decode the audio


    with torch.no_grad():
        torch.cuda.synchronize()
        time_start = time.time()
        for i in range(10):
            audio_path = "/home/gaochenghao/data/NiuTransData/data/zh/" + str(i + 1) + ".wav"
            print(audio_path)
            # origin
            result = model.transcribe(audio_path, temperature=0, initial_prompt="", word_timestamps=False)
            # prompt+word_timestamps
            #result = model.transcribe(audio_path, temperature=0, initial_prompt="需要简体中文，还有标点。", word_timestamps=True)
            print(result["text"])

        torch.cuda.synchronize()
        time_end = time.time()
        print("运行时间：{:.2f}ms".format((time_end - time_start) * 1000))
