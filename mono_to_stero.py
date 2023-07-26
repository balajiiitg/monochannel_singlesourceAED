import wave, array
import os
import glob
def make_stereo(file1, output):
    ifile = wave.open(file1)

    # (1, 2, 44100, 2013900, 'NONE', 'not compressed')
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = ifile.getparams()
    assert comptype == 'NONE'  # Compressed not supported yet
    array_type = {1:'B', 2: 'h', 4: 'l'}[sampwidth]
    left_channel = array.array(array_type, ifile.readframes(nframes))[::nchannels]
    ifile.close()

    stereo = 2 * left_channel
    stereo[0::2] = stereo[1::2] = left_channel

    ofile = wave.open(output, 'w')
    ofile.setparams((2, sampwidth, framerate, nframes, comptype, compname))
    ofile.writeframes(stereo.tostring())
    ofile.close()

# make_stereo("/home/balaji/Downloads/drums.wav", "drums.wav")

audio_files = glob.glob("/home/balaji/Documents/AED_research/seld-dcase2022-main/dataset/foa_dev/dev-test-sony/*.wav")
dst_path = "/home/balaji/Documents/AED_research/seld-dcase2022-main/data_set_mono/foa_dev/dev-test-sony/"
for file in audio_files:
    print(file)
    head_tail = os.path.split(file)
    print(head_tail)
    make_stereo(file, str(dst_path+head_tail[1]))
