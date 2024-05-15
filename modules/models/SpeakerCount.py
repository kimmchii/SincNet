import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
import numpy as np
import sys
import os.path as op
sys.path.append(op.dirname(__file__))
from SincNetBN import SincNetBN

class SpeakerCount(nn.Module):
    def __init__(self, config, device):
        super(SpeakerCount, self).__init__()
        self.device = device
        self.sincnet_bn_model = SincNetBN(config).to(self.device)

        # Freeze the sincnet_bn model
        for param in self.sincnet_bn_model.parameters():
            param.requires_grad = False

        speaker_counter_config = config["speaker_counter"]
        speaker_counter_config["input_dim"] = self.sincnet_bn_model.out_dim * 2

        self.speaker_counter_layers = self.init_speaker_counter(speaker_counter_config)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=speaker_counter_config["input_dim"], num_heads=speaker_counter_config["num_heads"], batch_first=True)

        self.window_len = int(config["CNN"]["fs"] * config["CNN"]["convolution_window_len"] / 1000.0)
        self.window_shift = int(
            config["CNN"]["fs"] * config["CNN"]["convolution_window_shift"] / 1000.0
        )
        self.batch_dev = config["optimizer"]["batch_dev"]
        self.d_vector_dim = self.sincnet_bn_model.out_dim

    def init_speaker_counter(self, config):
        return Sequential(
            nn.Linear(config["input_dim"], config["fc1_size"]),
            nn.Dropout(config["dropout_1"]),
            nn.Linear(config["fc1_size"], config["fc2_size"]),
            nn.Dropout(config["dropout_2"]),
            nn.Linear(config["fc2_size"], config["num_classes"]),
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
    
    def compute_vectors(self, signal):
        signal = signal / np.max(np.abs(signal))
        signal = torch.from_numpy(signal).float().to(self.device).contiguous()
        energy_array_bin = self.bypass_low_energy_frame(signal, self.window_len, self.window_shift)

        # split signals into chunks
        beging_sample = 0
        end_sample = self.window_len

        n_frame = int((signal.shape[0] - self.window_len) / (self.window_shift))

        signal_array = (
            torch.zeros([self.batch_dev, self.window_len]).float().to(self.device).contiguous()
        )
        d_vectors = (
            torch.zeros(n_frame, self.d_vector_dim).float().to(self.device).contiguous()
        )
        count_frame = 0
        count_frame_total = 0
        while end_sample < signal.shape[0]:
            signal_array[count_frame, :] = signal[beging_sample:end_sample]
            beging_sample = beging_sample + self.window_shift
            end_sample = beging_sample + self.window_len
            count_frame = count_frame + 1
            count_frame_total = count_frame_total + 1
            if count_frame == self.batch_dev:
                input = signal_array
                d_vectors[count_frame_total - self.batch_dev : count_frame_total, :] = (
                    self.sincnet_bn_model(input)
                )
                count_frame = 0
                signal_array = (
                    torch.zeros([self.batch_dev, self.window_len]).float().to(self.device).contiguous()
                )

        if count_frame > 0:
            input = signal_array[0:count_frame]
            d_vectors[count_frame_total - count_frame : count_frame_total, :] = (
                self.sincnet_bn_model(input)
            )

        
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
            return None
        return d_vector_out

    def forward(self, x):
        print(x.shape)
        x = self.compute_vectors(x)

        if x is None:
            return None

        x = x.unsqueeze(0)
        # Get the index 0 since it is the weight with attention.
        x = self.multihead_attn(x, x, x)[0]
        x = self.speaker_counter_layers(x)
        return F.softmax(x, dim=1)

if __name__ == "__main__":
    import yaml
    with open("../../models/speaker_classifcation/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = SpeakerCount(config, device)
    sincnet_statedict = torch.load("../../sincnet_models/sincnet_timit/train/cnn_state_dict.pth", map_location=model.device)
    dnn_statedict = torch.load("../../sincnet_models/sincnet_timit/train/dnn1_state_dict.pth", map_location=model.device)
    model.sincnet_bn_model.sincnet.load_state_dict(sincnet_statedict)
    model.sincnet_bn_model.dnn.load_state_dict(dnn_statedict)

    signal = np.random.rand(16000)
    print(model(signal))