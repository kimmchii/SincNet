# compute_d_vector.py
# Mirco Ravanelli
# Mila - University of Montreal

# Feb 2019

# Description:
# This code computes d-vectors using a pre-trained model


import os
import os.path as op
import soundfile as sf
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from modules.models.SincNet import MLP
from modules.models.SincNet import SincNetBN
from data_io import ReadList, read_conf_inp, str_to_bool
import sys
import yaml
from tqdm import tqdm


def compute_vectors(
    test_data_list: list[str],
    data_folder: str,
    avoid_small_energy_frame: bool,
    window_len: int,
    window_shift: int,
    d_vector_dim: int = 512,
    model: nn.Module = None,
):

    d_vect_dict = {}
    len_test = len(test_data_list)

    for i in tqdm(range(len_test)):

        [signal, fs] = sf.read(op.join(data_folder, test_data_list[i]))

        # Amplitude normalization
        signal = signal / np.max(np.abs(signal))

        signal = torch.from_numpy(signal).float().to(device).contiguous()

        if avoid_small_energy_frame:
            # computing energy on each frame:
            beging_sample = 0
            end_sample = window_len
            n_frame = int((signal.shape[0] - window_len) / (window_shift))
            batch_dev = n_frame
            energy_array = torch.zeros(n_frame).float().contiguous().to(device)
            count_frame = 0
            count_frame_total = 0
            while end_sample < signal.shape[0]:
                energy_array[count_frame] = torch.sum(
                    signal[beging_sample:end_sample].pow(2)
                )
                beging_sample = beging_sample + window_shift
                end_sample = beging_sample + window_len
                count_frame = count_frame + 1
                count_frame_total = count_frame_total + 1
                if count_frame == n_frame:
                    break

            energy_array_bin = energy_array > torch.mean(energy_array) * 0.1
            energy_array_bin.to(device)
            n_vect_elem = torch.sum(energy_array_bin)

            if n_vect_elem < 10:
                print("only few elements used to compute d-vectors")
                sys.exit(0)

        # split signals into chunks
        beging_sample = 0
        end_sample = window_len

        n_frame = int((signal.shape[0] - window_len) / (window_shift))

        signal_array = (
            torch.zeros([batch_dev, window_len]).float().to(device).contiguous()
        )
        d_vectors = Variable(
            torch.zeros(n_frame, d_vector_dim).float().to(device).contiguous()
        )
        count_frame = 0
        count_frame_total = 0
        while end_sample < signal.shape[0]:
            signal_array[count_frame, :] = signal[beging_sample:end_sample]
            beging_sample = beging_sample + window_shift
            end_sample = beging_sample + window_len
            count_frame = count_frame + 1
            count_frame_total = count_frame_total + 1
            if count_frame == batch_dev:
                input = Variable(signal_array)
                d_vectors[count_frame_total - batch_dev : count_frame_total, :] = (
                    model(input)
                )
                count_frame = 0
                signal_array = (
                    torch.zeros([batch_dev, window_len]).float().to(device).contiguous()
                )

        if count_frame > 0:
            input = Variable(signal_array[0:count_frame])
            d_vectors[count_frame_total - count_frame : count_frame_total, :] = (
                model(input)
            )

        if avoid_small_energy_frame:
            d_vectors = d_vectors.index_select(
                0, (energy_array_bin == 1).nonzero().view(-1)
            )

        # averaging and normalizing all the d-vectors
        d_vector_mean = torch.mean(
            d_vectors / d_vectors.norm(p=2, dim=1).view(-1, 1), dim=0
        )
        d_vect_variance = torch.var(d_vectors, dim=0)
        d_vect_out = torch.cat((d_vector_mean, d_vect_variance), 0)
        # checks for nan
        nan_sum = torch.sum(torch.isnan(d_vect_out))

        if nan_sum > 0:
            print(test_data_list[i])
            sys.exit(0)

        # saving the d-vector in a numpy dictionary
        # TODO: Check if this is the correct way to get the key, it's quite hardcoded.
        dict_key = (
            test_data_list[i].split("/")[-2] + "/" + test_data_list[i].split("/")[-1]
        )
        d_vect_dict[dict_key] = d_vect_out.cpu().numpy()
    return d_vect_dict


if __name__ == "__main__":
    with open("./sincnet_models/sincnet_timit/sincnet_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # [data]
    data_config = config["data"]
    test_data_list = ReadList(data_config["test"]["files"])
    len_test = len(test_data_list)
    out_dict_file = "./d_vector_timit.npy"
    avoid_small_energy_frame = True

    batch_dev = config["optimizer"]["batch_dev"]

    # Load state dict
    sincnet_state_dict = torch.load(
        "./sincnet_models/sincnet_timit/train/cnn_state_dict.pth"
    )
    dnn_state_dict = torch.load(
        "./sincnet_models/sincnet_timit/train/dnn1_state_dict.pth"
    )

    # [model]
    window_len = int(config["CNN"]["fs"] * config["CNN"]["convolution_window_len"] / 1000.0)
    window_shift = int(
        config["CNN"]["fs"] * config["CNN"]["convolution_window_shift"] / 1000.0
    )

    sincnet_bn = SincNetBN(config, device=device)
    sincnet_bn.sincnet.load_state_dict(sincnet_state_dict)
    sincnet_bn.dnn.load_state_dict(dnn_state_dict)

    # Inference
    sincnet_bn.eval()
    with torch.no_grad():
        d_vect_dict = compute_vectors(
            test_data_list = test_data_list,
            data_folder = data_config["test"]["data_folder"],
            avoid_small_energy_frame = avoid_small_energy_frame,
            window_len=window_len,
            window_shift=window_shift,
            model = sincnet_bn,
        )
        d_vect_dict = np.save(out_dict_file, d_vect_dict)
