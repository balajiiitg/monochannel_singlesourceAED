from pydub import AudioSegment as am
import os
import glob
# filepath ='/home/balaji/Downloads/ringtone.flac'
# sound = am.from_file(filepath, format='wav', frame_rate=24000)


def down_sample(file,  dest):
    sound = am.from_file(file, format='wav', frame_rate=24000)
    sound = sound.set_frame_rate(24000)
    sound.export(dest, format='wav')


audio_files = glob.glob("/home/balaji/Documents/Ajay_san_PhD_work/seld-dcase2022-main/dataset_1/temp1/*.wav")
dst_path = "/home/balaji/Documents/Ajay_san_PhD_work/seld-dcase2022-main/dataset_1/temp2/"
for file in audio_files:
    print(file)
    head_tail = os.path.split(file)
    print(head_tail)
    dest = str(dst_path+head_tail[1])
    down_sample(file, dest)