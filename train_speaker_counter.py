import os
import torch
from modules.datasets.speaker_count_dataset import LightningSpeakerCounterDataModule
from modules.models.SpeakerCount import LightningSpeakerCount
import yaml
import pytorch_lightning as pl

with open("./models/speaker_classifcation/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

# Load data module
train_data_path = "./dataset/speaker_count/metadata.json"
val_data_path = "./dataset/speaker_count/metadata.json"

speaker_count_data_module = LightningSpeakerCounterDataModule(train_data_path, val_data_path, batch_size=config["batch_size"])

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

speaker_count = LightningSpeakerCount(config=config, device=device)

# Train the model
trainer = pl.Trainer(max_epochs=1)
trainer.fit(speaker_count, speaker_count_data_module)



                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 









