import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
import numpy as np
import sys
import os.path as op
import pytorch_lightning as pl
from torchmetrics import F1Score as F1
import math

sys.path.append(op.dirname(__file__))
from SincNetBN import SincNetBN

class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CustomMultiheadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        attention_output, _ = self.attention(x, x, x)
        return attention_output

class SpeakerCount(nn.Module):
    """
    Speaker count model that uses SincNet and DNN to extract d-vectors and then uses a multihead attention mechanism
    to get the weights of the d-vectors and then uses a fully connected layer to get the speaker count.

    Args:
        config: dict
            Configuration dictionary.
        device: str
            Device to run the model on.
    
    Example:
    >>> import yaml
    >>> with open("../../models/speaker_classifcation/config.yaml", "r") as f:
    >>>     config = yaml.load(f, Loader=yaml.SafeLoader)
    >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> model = SpeakerCount(config, device)
    >>> signal = np.random.rand(16000)
    >>> print(model(signal))
    """
    def __init__(self, config):
        super(SpeakerCount, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sincnet_config = config["sincnet"]
        self.sincnet_bn_model = SincNetBN(sincnet_config)
        
        # Load the state dict
        sincnet_statedict = sincnet_config["cnn"].get("state_dict_path", None)
        dnn_statedict = sincnet_config["dnn"].get("state_dict_path", None)
        if sincnet_statedict is None:
            raise ValueError("SincNet state dict path is not provided.")
        if dnn_statedict is None:
            raise ValueError("DNN state dict path is not provided.")
        
        self.sincnet_bn_model.sincnet.load_state_dict(torch.load(sincnet_statedict, map_location=self.device))
        self.sincnet_bn_model.dnn.load_state_dict(torch.load(dnn_statedict, map_location=self.device))

        # Freeze the sincnet_bn model
        for param in self.sincnet_bn_model.parameters():
            param.requires_grad = False

        self.sincnet_bn_model.eval()

        speaker_counter_config = config["speaker_counter"]
        speaker_counter_config["input_dim"] = self.sincnet_bn_model.out_dim * 2

        self.speaker_counter_layers = self.init_speaker_counter(speaker_counter_config)
        # Initialize the weights of the speaker counter layers
        for m in self.speaker_counter_layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        # self.multihead_attn = nn.MultiheadAttention(embed_dim=speaker_counter_config["input_dim"], num_heads=speaker_counter_config["num_heads"], batch_first=True)

        self.window_len = int(sincnet_config["cnn"]["fs"] * sincnet_config["cnn"]["convolution_window_len"] / 1000.0)
        self.window_shift = int(
            sincnet_config["cnn"]["fs"] * sincnet_config["cnn"]["convolution_window_shift"] / 1000.0
        )
        self.batch_dev = sincnet_config["batch_dev"]
        self.d_vector_dim = self.sincnet_bn_model.out_dim

    def init_speaker_counter(self, config):
        return Sequential(
            # CustomMultiheadAttention(config["input_dim"], config["num_heads"]),
            nn.Linear(config["input_dim"], config["fc1_size"]),
            nn.ReLU(),
            nn.Dropout(config["dropout_1"]),
            nn.Linear(config["fc1_size"], config["fc2_size"]),
            nn.ReLU(),
            nn.Dropout(config["dropout_2"]),
            nn.Linear(config["fc2_size"], config["output_num_classes"]),
        )
    
    def bypass_low_energy_frame(self, signal, window_len, window_shift):
        beging_sample = 0
        end_sample = window_len
        n_frame = int((signal.shape[0] - window_len) / (window_shift))
        energy_array = torch.zeros(n_frame).float().contiguous().to(self.device)
        count_frame = 0
        count_frame_total = 0
        while end_sample < signal.shape[0]:
            energy_array[count_frame] = torch.sum(signal[beging_sample:end_sample].pow(2))
            beging_sample = beging_sample + window_shift
            end_sample = beging_sample + window_len
            count_frame = count_frame + 1
            count_frame_total = count_frame_total + 1
            if count_frame == n_frame:
                break
        energy_array_bin = energy_array > torch.mean(energy_array) * 0.1
        energy_array_bin.to(self.device)
        n_vect_elem = torch.sum(energy_array_bin)
        if n_vect_elem < 10:
            return None
        return energy_array_bin
    
    def compute_vectors(self, signals):
        d_vector_outs = []
        for signal in signals:
            # Change the dimension of the signal from (channels, samples) to (samples,)
            signal = signal.cpu()
            signal = signal.squeeze().numpy()
            signal = signal / np.max(np.abs(signal))
            signal = torch.from_numpy(signal).float().contiguous().to(self.device)
            energy_array_bin = self.bypass_low_energy_frame(signal, self.window_len, self.window_shift)

            # split signals into chunks
            beginning_sample = 0
            ending_sample = self.window_len

            # n_frame = int((signal.shape[0] - self.window_len) // self.window_shift)

            # Round up to the nearest integer to avoid losing frames in the case that the signal length is not a multiple of the window shift.
            n_frame = math.ceil((signal.shape[0] - self.window_len) / self.window_shift)

            signal_array = (
                torch.zeros([self.batch_dev, self.window_len]).float().contiguous().to(self.device)
            )
            d_vectors = (
                torch.zeros(n_frame, self.d_vector_dim).float().contiguous().to(self.device)
            )
            count_frame = 0
            count_frame_total = 0

            while ending_sample < signal.shape[0]:
                signal_array[count_frame, :] = signal[beginning_sample: ending_sample]
                beginning_sample += self.window_shift
                ending_sample = beginning_sample + self.window_len
                count_frame += 1
                count_frame_total += 1
                if count_frame == self.batch_dev:
                    input = signal_array
            
                    with torch.no_grad():
                        d_vectors[count_frame_total - self.batch_dev : count_frame_total, :] = self.sincnet_bn_model(input)
                    
                    count_frame = 0
                    signal_array = (
                        torch.zeros([self.batch_dev, self.window_len]).float().contiguous().to(self.device)
                    )
            
            if count_frame > 0:
                input = signal_array[0: count_frame]
                
                with torch.no_grad():
                    d_vectors[count_frame_total - count_frame: count_frame_total, :] = self.sincnet_bn_model(input)

            d_vectors = d_vectors.index_select(
                0, (energy_array_bin == 1).nonzero().view(-1)
            )

            # averaging and normalizing all the d-vectors
            d_vector_mean = torch.mean(
                d_vectors / d_vectors.norm(p=2, dim=1).view(-1, 1), dim=0
            )
            d_vect_variance = torch.var(d_vectors, dim=0)
            d_vector_out = torch.cat((d_vector_mean, d_vect_variance), 0)
            # checks for nan
            nan_sum = torch.sum(torch.isnan(d_vector_out))
            if nan_sum > 0:
                print("Nan in d-vectors")
                d_vector_out = torch.zeros(self.d_vector_dim * 2).to(self.device)
                
            d_vector_outs.append(d_vector_out)
        
        d_vector_outs = torch.stack(d_vector_outs)
        return d_vector_outs

    def forward(self, x):
        # x dimension is (batch_size, channels, samples), output dimension is (batch_size, samples)
        x = self.compute_vectors(x)
        x = self.speaker_counter_layers(x)
        return x
        
class LightningSpeakerCount(pl.LightningModule):
    """
    Pytorch Lightning module for the speaker count model.

    Args:
        config: dict
            Configuration dictionary.
        device: str
            Device to run the model on.
    
    Example:
    >>> import yaml
    >>> with open("../../models/speaker_classifcation/config.yaml", "r") as f:
    >>>     config = yaml.load(f, Loader=yaml.SafeLoader)
    >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    def __init__(self, config):
        super(LightningSpeakerCount, self).__init__()
        self.model = SpeakerCount(config)
        self.learning_rate = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.loss = nn.CrossEntropyLoss()
        seed = 42
        torch.manual_seed(seed)
        self.f1 = F1(task="multiclass", num_classes=config["speaker_counter"]["output_num_classes"])


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        predicted_output = torch.argmax(y_hat, dim=1)
        self.log("F1 score", self.f1(predicted_output, y))
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":
    import yaml
    with open("../../models/speaker_classifcation/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeakerCount(config, device)
    model.to(device)
    signal = np.random.rand(6401)
    signal = torch.tensor(signal).unsqueeze(0)
    batch_signal = torch.stack([torch.tensor(signal) for _ in range(2)])
    print(batch_signal.shape)
    print(model(batch_signal))
