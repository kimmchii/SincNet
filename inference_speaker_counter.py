import torch
from modules.models.SpeakerCount import LightningSpeakerCount
import yaml
import librosa

if __name__ == "__main__":
    with open("./models/speaker_classifcation/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    speaker_count = LightningSpeakerCount(config=config)

    # Load the speaker counter layers
    speaker_count_state_dict = torch.load("./models/speaker_classifcation/speaker_count/speaker_counter_layers.pth")
    speaker_count.model.speaker_counter_layers.load_state_dict(speaker_count_state_dict)

    audio_path = "./dataset/speaker_count/3/audio_35477.mp3"
    # audio_path = "./dataset/speaker_count/1/si1005.mp3"
    # audio_path = "./dataset/speaker_count/2/audio_20000.mp3"
    audio_array, sr = librosa.load(audio_path, sr=8000)
    speaker_count.to(device)

    # Inference the model
    with torch.no_grad():
        speaker_count.eval()

        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).unsqueeze(0).to(device)
        outputs = speaker_count(audio_tensor)
        print(torch.softmax(outputs, dim=1))
        # num_speaker = torch.argmax(outputs)
    # print(num_speaker)
    

