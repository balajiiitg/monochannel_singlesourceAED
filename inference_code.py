
import numpy as np
import librosa
import torch
import numpy as np
import os
import sys

from torch._C import device

import cls_data_generator
import seldnet_model
import parameters
import torch
from IPython import embed
import matplotlib
def main(argv):


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    print('\nLoading the best model and predicting results on the testing split')
    print('\tLoading testing dataset:')
    data_gen_test = cls_data_generator.DataGenerator(
        params=params, split=4, shuffle=False, is_eval=True if params['mode']=='eval' else False
    )
    data_in, data_out = data_gen_test.get_data_sizes()
    dump_figures = True

    # CHOOSE THE MODEL WHOSE OUTPUT YOU WANT TO VISUALIZE
    checkpoint_name = "/home/noesis/workspace_balaji/workspace_priya/seld-dcase2022-main/models/1_1_dev_split0_accdoa_mic_gcc_model.h5"
    model = seldnet_model.CRNN(data_in, data_out, params)
    model.eval()
    model.load_state_dict(torch.load(checkpoint_name, map_location=torch.device('cpu')))
    model = model.to(device)

    return model, device
# Load the trained model


# Define the classes
classes = ['Female speech women speaking', 'Male speech man speaking', 'Clapping', 'Telephone', 'Laughter', 'Domestic sounds',
           'Walk foot steps', 'Door open or door close', 'Music', 'Musical instrument', 'Water tap', 'Bell', 'Knock']  # Replace with your own class names

def load_model():
    checkpoint_name = "/home/noesis/workspace_balaji/workspace_priya/seld-dcase2022-main/models/1_1_dev_split0_accdoa_mic_gcc_model.h5"

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # use parameter set defined by user
    task_id = '1'
    params = parameters.get_params(task_id)
    data_gen_test = cls_data_generator.DataGenerator(
        params=params, split=4, shuffle=False, is_eval=True if params['mode'] == 'eval' else False
    )
    data_in, data_out = data_gen_test.get_data_sizes()
    model = seldnet_model.CRNN(data_in, data_out, params)
    model.eval()
    model.load_state_dict(torch.load(checkpoint_name, map_location=torch.device('cpu')))
    model = model.to(device)
    return  model
# Function to preprocess the audio data
def preprocess_audio(audio_file):

    # Load the audio file
    audio, sr = librosa.load(audio_file, sr=None)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # Convert the audio to mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(audio, sr=sr)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram = mel_spectrogram[np.newaxis, np.newaxis, ...]
    mel_spectrogram = torch.from_numpy(mel_spectrogram).float().to(device)

    return mel_spectrogram


# Function to perform sound event detection on a single audio file



def detect_sound_event(audio_file):
    # Preprocess the audio data
    mel_spectrogram = preprocess_audio(audio_file)

    # Perform inference using the trained model
    with torch.no_grad():
        model= load_model()
        print("mmm", mel_spectrogram)
        output = model(mel_spectrogram)

        predictions = torch.softmax(output, dim=1).cpu().numpy()

    predicted_class = np.argmax(predictions)
    predicted_prob = np.max(predictions)

    # Get the predicted class label
    predicted_label = classes[predicted_class]

    return predicted_label, predicted_prob


# Example usage
audio_file = '/home/noesis/workspace_balaji/workspace_priya/seld-dcase2022-main/dataset/drums.wav'  # Replace with your own audio file
predicted_label, predicted_prob = detect_sound_event(audio_file)

# Print the predicted class label and probability
print(f'Predicted Label: {predicted_label}')
print(f'Predicted Probability: {predicted_prob}')

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)