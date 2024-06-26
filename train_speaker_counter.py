import torch
from modules.datasets.speaker_count_dataset import LightningSpeakerCounterDataModule
from modules.models.SpeakerCount import LightningSpeakerCount
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping


with open("./models/speaker_classifcation/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

# Load data module
train_data_path = "./dataset/speaker_count/metadata_train.json"
val_data_path = "./dataset/speaker_count/metadata_val.json"

speaker_count_data_module = LightningSpeakerCounterDataModule(train_data_path, val_data_path, batch_size=config["batch_size"])

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

speaker_count = LightningSpeakerCount(config=config).to(device)

# Train the model
wandb_logger = WandbLogger(project="Speaker Count")

# Train the model
trainer = pl.Trainer(
                    max_epochs=config["n_epoch"],
                    gradient_clip_val=5,
                    callbacks=[
                        ModelCheckpoint(
                            monitor="val_loss",
                            mode="min",
                            save_top_k=1,
                            dirpath="./models/speaker_classifcation/speaker_count",
                            filename="best_model"
                        ),
                        EarlyStopping(
                            monitor="val_loss",
                            mode="min",
                            patience=10,
                        )
                    ],
                    precision="16-mixed",
                    accelerator="auto",
                    logger=wandb_logger,
                    )
trainer.fit(
    model=speaker_count,
    datamodule=speaker_count_data_module
    )


# Save speaker_counter_layers to be used in the speaker classification model
torch.save(speaker_count.model.speaker_counter_layers.state_dict(), "./models/speaker_classifcation/speaker_count/speaker_counter_layers.pth")




                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 









