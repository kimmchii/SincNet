import torch
import yaml
import librosa

from modules.models.SpeakerCount import LightningSpeakerCount

state_dict = torch.load("./models/speaker_classifcation/speaker_count/best_model.ckpt")

with open("./models/speaker_classifcation/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

model = LightningSpeakerCount(config)

model.load_state_dict(state_dict["state_dict"])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# audio_path = "./dataset/speaker_count/3/audio_35477.mp3"
# audio_path = "./dataset/speaker_count/1/si1005.mp3"
# audio_path = "./dataset/speaker_count/2/audio_20000.mp3"
# audio_path = "./dataset/speaker_count/0/1-100032-A-0.wav"
# audio_path = "./crop/cut_mhm_0.mp3"
# audio_path = "./crop/cut_mhm.mp3"
audio_path = "./crop/cropped_audio.wav"
audio_array, sr = librosa.load(audio_path, sr=8000)
# pad the audio array to 8000 samples
model.to(device)

model.eval()

with torch.no_grad():
    output = model(torch.tensor(audio_array).to(device).unsqueeze(0))
    # print(output)
    print(torch.softmax(output, dim=1))
