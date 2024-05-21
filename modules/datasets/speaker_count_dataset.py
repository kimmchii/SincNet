import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import pad
import pytorch_lightning as pl
import librosa
import json

class SpeakerCountDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = self.data[idx]["path"]
        label = int(self.data[idx]["num_speaker"])
        audio_array, _ = librosa.load(data_path, sr=8000)
        audio_tensor = torch.from_numpy(audio_array)
        label_tensor = torch.tensor(label)
        return {"audio": audio_tensor, "label": label_tensor}
    
class LightningSpeakerCounterDataModule(pl.LightningDataModule):
    def __init__(self, train_data_path: str, val_data_path: str, batch_size: int=1):
        super().__init__()
        self.batch_size = batch_size

        with open(train_data_path, "r") as f:
            self.train_data = json.load(f)

        with open(val_data_path, "r") as f:
            self.val_data = json.load(f)

    def custom_collate_fn(self, batch):
        """
        Customs collate function to pad the audio to the maximum length.
        Inputs:
            batch: List of dictionaries containing the merged audio and the target audios.
        
        Returns:
            merged_audios:
                Tensors of merged audios padded to the maximum length.

            targets:
                Tensors of target audios padded to the maximum length.
        """
        audios = [item["audio"] for item in batch]
        max_len = max([len(audio) for audio in audios])
        pad_audios = [pad(audio, (0, max_len - len(audio))).unsqueeze(0) for audio in audios]

        labels = [item["label"] for item in batch]
        audio_tensors = torch.stack(pad_audios, dim=0)
        label_tensors = torch.stack(labels)

        # Check nan values
        if torch.isnan(audio_tensors).any():
            raise ValueError("NaN values in audio tensors")
        return audio_tensors, label_tensors

    def setup(self, stage=None):
        self.train_data_tensors = SpeakerCountDataset(self.train_data)
        self.val_data_tensors = SpeakerCountDataset(self.val_data)

    def train_dataloader(self):
        # Deterministic shuffle
        return DataLoader(self.train_data_tensors, batch_size=self.batch_size, collate_fn=self.custom_collate_fn, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data_tensors, batch_size=self.batch_size, collate_fn=self.custom_collate_fn)
