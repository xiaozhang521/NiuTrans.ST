import time

import whisper

from subprocess import CalledProcessError, run
import numpy as np
import torch


def load_audio(file: str, sr=16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "/home/gaochenghao/tools/ffmpeg-6.0.1/ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

if __name__ == '__main__':
    # path="/home/gaochenghao/data/whisper_model"
    # model_type="tiny"



    model = whisper.load_model("tiny")

    # load audio and pad/trim it to fit 30 seconds
    audio = load_audio("/data/zhangyuhao/librispeech/LibriSpeech/test-clean/672/122797/672-122797-0000.flac")
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    options = whisper.DecodingOptions()

    # decode the audio
    time_start = time.time()
    torch.cuda.synchronize()
    for i in range(10):
        result = whisper.decode(model, mel, options)
        # print the recognized text
        print(result.text)
    torch.cuda.synchronize()
    time_end= time.time()
    print("运行时间：{:.2f}ms".format((time_end - time_start)*1000))