import os, sys
import torch
import webrtcvad
import numpy as np
import librosa
import wave
import contextlib
import soundfile as sf
import time
import collections
import argparse
import math
import torch.nn.functional as F

from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union



# whisper 

SAMPLE_RATE = 16000     # 采样率
N_FFT = 400             # 窗口大小
N_MELS = 80             # fbank通道数

def load_audio(file: str, sr: int = SAMPLE_RATE):

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
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

@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(
        os.path.join(os.path.dirname(__file__),  "mel_filters.npz")
    ) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

# self

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration, num, b):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration
        self.num_bytes = num
        self.b = b
        self.e = b + num - 1
    def show(self):
        print("timestamp: {:.4f}, duration: {}, bytes: {} [{}~{}].".format(self.timestamp, self.duration, self.num_bytes, self.s, self.e))

def frame_generator(audio, frame_duration_ms: float=30.0):
    global sr
    
    n = int(sr * (frame_duration_ms / 1000.0) * 2)

    # print("[Info] [frame_generator] Frame length {} s ({} bytes).".format(frame_duration_ms / 1000, n))

    offset = 0
    timestamp = 0.0
    frames = []
    while offset + n < len(audio):
        audio_segement = audio[offset : offset + n]
        num_audio_bytes = len(audio_segement)
        duration = (float(num_audio_bytes) / sr) / 2.0
        frame = Frame(audio_segement, timestamp, duration, num_audio_bytes, offset)
        # frames += [frame]
        yield frame
        
        timestamp += duration
        offset += n
    # return frames

def vad(vad_modal, audio, frame_duration_ms: float=30.0, padding_duration_ms: float=300.0, padding_rate=0.9):
    global sr

    # bytes
    num_audio_bytes = len(audio)
    num_padding_bytes = int(sr * (padding_duration_ms / 1000.0) * 2)

    # second
    audio_duration_s = (float(len(audio)) / sr) / 2.0

    print("[Info] [vad] Audio length {} s ({} bytes), Padding length {} s ({} bytes).".format(audio_duration_s, num_audio_bytes, padding_duration_ms / 1000.0, num_padding_bytes))

    # make frame
    frames = frame_generator(audio, frame_duration_ms)
    frames = list(frames)
    
    # frames
    num_audio_frames = len(frames)
    assert padding_duration_ms % frame_duration_ms == 0
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)

    # buffer
    ring_buffer = collections.deque(maxlen=num_padding_frames)

    triggered = False
    voiced_frames = []

    num_frames = 0
    for frame in frames:
        try:
            if len(frame.bytes) != 960:
                print(len(frame.bytes))
            is_speech = vad_modal.is_speech(frame.bytes, sr)
        except :
            assert False
        
        # sys.stdout.write(str(num_frames))
        # sys.stdout.write(': 1  ' if is_speech else ': 0  ')
        # sys.stdout.write(str(triggered))

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            
            if num_voiced > padding_rate * ring_buffer.maxlen:
                triggered = True

                for f, s in ring_buffer:
                    voiced_frames.append(f)
                    # sys.stdout.write(" Add frames! ")
                ring_buffer.clear()

        else:
            voiced_frames.append(frame)
            # sys.stdout.write(" Add frames! ")

            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])

            if num_unvoiced > padding_rate * ring_buffer.maxlen:
                triggered = False

                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
        num_frames += 1

        # sys.stdout.write(str("\n"))

    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])

def split_with_length(audios, max_num_bytes):
    global sr

    new_audio = b''
    length = 0
    
    for a in audios:
        audio_length = len(a)
        if audio_length > max_num_bytes:
            assert False, "TODO!"
        new_length = length + audio_length
        if new_length < max_num_bytes:
            new_audio += a 
            length = new_length
        else:
            yield new_audio
            # print("[Info] [split] New audio length: {}".format(length))
            new_audio = a 
            length = audio_length
    if new_audio != b'':
        yield new_audio
        # print("[Info] [split] New audio length: {}".format(length))
        new_audio = b'' 
        length = 0

def load_pcm(path):
    global sr

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", path,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return out, sr

def featureExact(audio, padding: int = 3000, mel_filter_file: str = None):

    global fbank, sr, n_fft, hop_length

    num_padding_sample = padding * hop_length
    num_audio_sample = audio.shape[0]
    num_real_frames = num_audio_sample / hop_length

    print("[Info] [featureExact] Audio length {} s, samples {}, frames {}.".format((float(num_audio_sample) / sr), num_audio_sample, num_real_frames))
    print("[Info] [featureExact] Padding length {} s, samples {}, frames {}.".format((float(num_padding_sample) / sr), num_padding_sample, padding))

    # padding
    if padding > 0 and num_padding_sample > num_audio_sample:
        audio = F.pad(audio, (0, ((num_padding_sample - num_audio_sample) + (int(10) * hop_length))))

    window = torch.hann_window(n_fft)
    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    # mel_filters
    if mel_filter_file is None:
        filters = torch.from_numpy(librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=fbank))
        assert torch.equal(filters, mel_filters(audio.device, fbank))
    else:
        all_filters = np.load(mel_filter_file)
        filters = torch.from_numpy(all_filters["mel_{}".format(fbank)])
        # assert torch.equal(filters, mel_filters(audio.device, fbank))
        # assert torch.equal(filters, torch.from_numpy(librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=fbank)).to(audio.device)),"[Error] Wrong mel filter! {} : {}".format(mel_filter_file, fbank)

    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return  log_spec[:, :padding], int(min(num_real_frames, padding))

def saveFbankBinary(x, fbank, file):

    print("[Info] Saving {} Data {} in binary .".format(file, x.shape))

    assert len(x.shape) == 2, "[Error] Wrong shape!"
    if x.shape[1] != fbank:
        assert x.shape[0] == fbank, "[Error] No dim with 80 size!"
        x = x.transpose(0, 1)
    
    from struct import pack, unpack

    with open(file, 'wb') as f:
        values = pack("f" * x.numel(), * (x.contiguous().view(-1).cpu().numpy().astype(np.float32)))
        f.write(values)
    
    # print("[Info] Saved {} Data.".format(file))

fbank = 80
sr = 16000
n_fft = 400
hop_length = 160

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

def main(audios, output, save_path, vad_mode, duration:float =30.0):

    workspace = os.path.dirname(os.path.abspath(__file__))

    global fbank, sr, device
    assert fbank == 80, "only 80 fbank is supported"
    
    need_frames = int(duration * sr / hop_length)
    print("[Info] Process all audio to {} s frames {}.".format(duration, need_frames))
    
    with open(output, 'w') as output_f:

        output_f.write("id\taudio\tframes\ttgt_text\n")

        for index, audio_info in enumerate(audios):
            ids, audio, num_frames, transcribe = audio_info
            audio_name = os.path.basename(audio)
        
            # mel_filter_file = os.path.join(os.path.dirname(__file__),  "mel_filters.npz")

            # audio data
            print("[Info] Processing Audio: {}".format(audio))
            audio, _ = load_pcm(audio)

            max_num_bytes = duration * sr * 2
            if max_num_bytes > len(audio) and False:
                if vad_mode == 0:
                    # vad
                    vad_modal = webrtcvad.Vad(vad_mode)
                    audios = vad(vad_modal, audio, padding_rate=0.9)
                    audios = list(audios)
                else:
                    audios = [audio]
                
                # split
                audios = split_with_length(audios, max_num_bytes)
                audios = list(audios)
            else:
                audios = [audio]

            # fbank
            mels = []
            num_audio = 0

            print("[Info] num of audio segements: ", len(audios))
            for a in audios:
                num_audio += 1
                a_name = audio_name + "_{}.bin".format(num_audio)
                a = torch.from_numpy(np.frombuffer(a, dtype=np.int16).flatten().astype(np.float32) / 32768.0)
                mel_segment, frames = featureExact(a, padding=need_frames, mel_filter_file=None)
                print("[Info] audio: {}, shape: {}, frames: {}".format(a_name, mel_segment.shape, frames))

                saveFbankBinary(mel_segment, 80, os.path.join(save_path, a_name))
                # output_f.write("\t".join([ids, os.path.join(save_path, a_name), str(frames), transcribe]) + "\n")
                if transcribe is not None:

                    output_f.write("\t".join([str(index), os.path.join(save_path, a_name), str(mel_segment.shape[1]), transcribe]) + "\n")
                else:
                    output_f.write("\t".join([str(index), os.path.join(save_path, a_name), str(mel_segment.shape[1]), ""]) + "\n")
                
    
    

if __name__ == "__main__":

    start = time.time()

    parser = argparse.ArgumentParser(prog='Audio Exactor', description='Read audio files, VAD and exact fbank feature')
    
    parser.add_argument("-a", "--audio", help="Path to Audio file", type=str)
    parser.add_argument("-l", "--list", help="Tsv file include all audio file path, (id\taudio\tframes\ttgt_text)", type=str)
    parser.add_argument("-p", "--path", help="Path to save audio file(bin)", type=str)
    parser.add_argument("-o", "--output", help="Tsv file include list of new audio, (id\taudio\tframes\ttgt_text)", type=str)
    parser.add_argument("-v", "--vad", help="Choose vad strength, not use vad when 0", type=int)
    parser.add_argument("-d", "--duration", help="Duration needs pad to", type=float, default=30.0)

    args = parser.parse_args()

    # print(args)
    
    # get audio file list
    audios = []
    if args.audio:
        audios.append([None, args.audio, None, None])
    if args.list:
        with open(args.list, 'r') as l:
            contents = l.read().split('\n')

        # print(contents[0])
        assert contents[0] == "id\taudio\tframes\ttgt_text"
        
        for line in contents[1:]:
            if line:
                audios.append(tuple(line.split('\t')))
    
    main(audios, os.path.abspath(args.output), os.path.abspath(args.path), args.vad, args.duration)

    end = time.time()

    print("[Info] Using time: {} s.".format(end - start))