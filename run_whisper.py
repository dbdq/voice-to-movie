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
        print(">> Using CUDA")
        model = whisper.load_model(model, device="cuda")
    elif device == 'mps' and torch.backends.mps.is_available():
        print(">> Using MPS")
        from lightning_whisper_mlx import LightningWhisperMLX
        model = LightningWhisperMLX(model=model, batch_size=12, quant=None)
    else:
        print(">> Using CPU")
        model = whisper.load_model(model, device="cpu")

    print(f'\n>> Transcribing {audio_file}.')
    t0 = time.time()
    result = model.transcribe(audio_file, language='ko')
    print('Took %d seconds.' % (time.time() - t0))
    # print(result["text"])
    qc.save_obj(output_file, result)
    print(f'Transcription saved to {output_file}.')
    return


def make_video(audio_file, trans_file, video_file):
    # encoding setting
    preset = "placebo"
    bitrate = "32k"
    video_duration = 1  # seconds (initial length)
    video_size = (1280, 720)  # width, height
    origianl_audio = True  # use original audio instead of compressing
    background_color = (0, 0, 0)  # black background
    fontsize = 45
    fps = 10

    # time measurement
    t0 = time.time()

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
                # graceful linefeed
                c = '\n'
                count = 0
            elif count > 30:
                # forced linefeed
                c += '\n'
                count = 0
            text_out.append(c)
            count += 1
        subtitle = mp.TextClip(text=''.join(text_out), font="MaruBuri-Regular.ttf", font_size=fontsize, color='white', size=video_size).with_position('left').with_start(start).with_duration(duration)
        subtitles.append(subtitle)

    # combine subtitles with blank video
    blank_video = mp.ColorClip(
        size=video_size, color=background_color, duration=video_duration
    )
    final_video = mp.CompositeVideoClip([blank_video] + subtitles)

    # export our final video
    if origianl_audio:
        final_video.write_videofile(
            video_file,
            fps=fps,
            remove_temp=False,
            codec="libx265",
            bitrate=bitrate,
            preset=preset,
            audio=audio_file,
        )
    else:
        # load audio and transcode to a specific quality
        audio_bitrate = "20k"
        audio_fps = 12000
        audio_codec = "aac"

        # export our video
        final_video.audio = mp.AudioFileClip(audio_file)
        final_video.write_videofile(
            video_file,
            fps=fps,
            remove_temp=False,
            codec="libx265",
            bitrate=bitrate,
            preset=preset,
            audio_codec=audio_codec,
            audio_bitrate=audio_bitrate,
            audio_fps=audio_fps,
        )

    print(f'Took %ds. Exported to {video_file}' % (time.time()-t0))


if __name__ == '__main__':
    # warning: 'mps' may result in poor quality
    processor = 'cuda' # cuda | mps | cpu
    model = 'large-v2'
    in_dir = r"/Users/leeq/Downloads/whisper/input"
    out_dir = f'{in_dir}/video'
    overwrite_video = True

    qc.make_dirs(out_dir)
    for audio_file in qc.get_file_list(in_dir):
        fname = qc.parse_path(audio_file).name
        trans_file = f'{out_dir}/{fname}.pkl'
        video_file = f'{out_dir}/{fname}.mkv'
        if not os.path.exists(trans_file):
            run_whisper(audio_file, trans_file, model, processor)
        if not os.path.exists(video_file) or overwrite_video:
            make_video(audio_file, trans_file, video_file)
