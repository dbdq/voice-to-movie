"""
Transcribe voice using Whisper and create a video with caption.

Models:
tiny
base
small
medium
large
large-v2
large-v3
turbo

Note:
Mac's Metal MPS GPU is not yet supported by pytorch, so
Lightning-Whisper should be used, but performance is worse.
https://github.com/mustafaaljadery/lightning-whisper-mlx

Author: Kyuhwa Lee

"""

import os
import time
import torch
import whisper
import numpy as np
import q_common as qc
import moviepy as mp

def run_whisper(audio_file, output_file, model='large_v2', device='cpu'):
    print(f'\n>> Loading {model} model.')
    if device == 'cuda' and torch.cuda.is_available():
        model = whisper.load_model(model, device='cuda')
        print('>> Using CUDA')
    elif device == 'mps' and torch.backends.mps.is_available():
        from lightning_whisper_mlx import LightningWhisperMLX
        model = LightningWhisperMLX(model=model, batch_size=12, quant=None)
        print('>> Using MPS')
    else:
        model = whisper.load_model(model, device='cpu')
        print('>> Using CPU')

    print(f'\n>> Transcribing {audio_file}.')
    t0 = time.time()
    result = model.transcribe(audio_file, language='ko')
    print('Took %d seconds.' % (time.time() - t0))
    #print(result["text"])
    qc.save_obj(output_file, result)
    print(f'Transcription saved to {output_file}.')
    return


def make_video(audio_file, trans_file, video_file):
    # video setting
    video_duration = 100 # seconds (initial length)
    video_size = (1280, 720) # width, height
    background_color = (0, 0, 0) # black background
    fps = 10
    fontsize = 45

    # load audio and create a blank video
    audioclip = mp.AudioFileClip(audio_file)
    blank_video = mp.ColorClip(size=video_size, color=background_color, duration=video_duration)

    # load transcription with timings
    text_segs = qc.load_obj(trans_file)['segments']
    subtitles = []
    for i, seg in enumerate(text_segs):
        start = seg['start']
        duration = seg['end'] - seg['start']
        text = seg['text'].strip()
        text_out = []
        count = 0
        for c in text:
            if count > 25 and c == ' ':
                c = '\n'
                count = 0
            text_out.append(c)
            count += 1
        subtitle = mp.TextClip(text=''.join(text_out), font="MaruBuri-Regular.ttf", font_size=fontsize, color='white', size=video_size).with_position('left').with_start(start).with_duration(duration)
        subtitles.append(subtitle)
        #if i > 10: break

    # combine subtitles with blank video
    final_video = mp.CompositeVideoClip([blank_video] + subtitles)
    final_video.audio = audioclip
    final_video.write_videofile(video_file, fps=fps, remove_temp=True, codec="libx265", audio_codec="aac", threads=0)
    print(f'Exported to {video_file}')


if __name__ == '__main__':
    processor = 'cuda' # cuda | mps | cpu
    model = 'large-v2'
    in_dir = r'C:\Users\leeq\Downloads\낭독\input'

    out_dir = f'{in_dir}/video'
    qc.make_dirs(out_dir)
    for audio_file in qc.get_file_list(in_dir):
        fname = qc.parse_path(audio_file).name
        trans_file = f'{out_dir}/{fname}.pkl'
        video_file = f'{out_dir}/{fname}.mkv'
        if not os.path.exists(trans_file):
            run_whisper(audio_file, trans_file, model, processor)
        #if not os.path.exists(video_file):
        make_video(audio_file, trans_file, video_file)
