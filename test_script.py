import sys
import os
import fnmatch
import moviepy.editor as mp
import matplotlib
import cv2
from ffpyplayer.player import MediaPlayer
import scipy.io.wavfile
import numpy as np

sys.path.append(os.path.abspath('external/audio-sync-kit'))
import audio_sync

sys.path.append(os.path.abspath('external/allison-deal-video-sync'))
import alignment_by_row_channels 


TEST_FILE_DIR = 'test-clips/greta-van-fleet'
VIDEO_TYPES = ['*.mp4']
AUDIO_TYPE = '.wav'
TMP_DIR = 'tmp'
if not os.path.exists(TMP_DIR):
    os.mkdir(TMP_DIR)

print('loading captures...')


videos = []
audio_paths = []
for fn in os.listdir(TEST_FILE_DIR):
    if any([fnmatch.fnmatch(fn, t) for t in VIDEO_TYPES]):
        print(f'loading {fn}')

        full_path = os.path.join(TEST_FILE_DIR, fn)
        clip = mp.VideoFileClip(full_path)
        videos.append(clip)

        # Save to temp audio file
        audio_fn = os.path.splitext(fn)[0] + AUDIO_TYPE
        audio_path = os.path.join(TMP_DIR, audio_fn)
        if os.path.exists(audio_path):
            print('Audio File already saved')
        else:
            print('Saving to temp path')
            clip.audio.write_audiofile(audio_path)
        audio_paths.append(audio_path)


import matplotlib.pyplot as plt

plt.figure()

for a in audio_paths:
    rate, data = scipy.io.wavfile.read(a)
    freq_data = np.fft.rfft(data)
    
    cutoff = int(len(freq_data) * 0.80)
    freq_data[cutoff:] = 0
    filtered_data = np.fft.irfft(freq_data)

    plt.plot(filtered_data[:rate*10])
    print(a)

plt.show()


# Compute the latencies
print('Calculating latencies...')
latencies, dropouts  = audio_sync.AnalyzeAudios(audio_paths[0], audio_paths[1])

print(f'Loaded {len(videos)} files...')
(a, b) = alignment_by_row_channels.align_from_audio(audio_paths[0], audio_paths[1])
print(a, b)

cap = cv2.VideoCapture('chaplin.mp4')
