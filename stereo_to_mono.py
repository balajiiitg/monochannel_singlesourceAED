from pydub import AudioSegment
import wave, array
import os
import glob

def make_mono(src_path, dst_path):
    sound = AudioSegment.from_wav(src_path)
    sound = sound.set_channels(1)
    sound.export(dst_path, format="wav")


audio_files = glob.glob("/home/balaji/Documents/AED_research/seld-dcase2022-main/dataset/mic_dev/dev-train-tau/*.wav")
dst_path = "/home/balaji/Documents/AED_research/seld-dcase2022-main/data_set_mono/mic_dev/dev-train-tau/"
for file in audio_files:
    print(file)
    head_tail = os.path.split(file)
    print(head_tail)
    make_mono(file, str(dst_path+head_tail[1]))
