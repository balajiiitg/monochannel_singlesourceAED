import librosa
import matplotlib.pyplot as plt

# Load the audio file
audio_file = '/home/noesis/Priya_work/seld-dcase2022-main/dataset/foa_dev/dev-test-sony/fold4_room23_mix001.wav'
audio_data, sr = librosa.load(audio_file)

# Perform event detection and obtain labels
# ...

# Plot the waveform with labels
plt.figure(figsize=(10, 4))
plt.plot(audio_data, label='Waveform')
plt.vlines(x='/home/noesis/Priya_work/seld-dcase2022-main/dataset/feat_label/foa_dev/fold4_room23_mix001.npy', ymin=-1, ymax=1, color='r', linewidth=2, label='Event')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title('Detected Events in Audio')
plt.legend()
plt.show()
