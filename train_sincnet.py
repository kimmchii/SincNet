# speaker_id.py
# Mirco Ravanelli
# Mila - University of Montreal

# July 2018

# Description:
# This code performs a speaker_id experiments with SincNet.

# How to run it:
# python speaker_id.py --cfg=cfg/SincNet_TIMIT.cfg

import os
import os.path as op

import soundfile as sf
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from dnn_models import MLP, SincNet
from data_io import ReadList
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_batches_rnd(
    batch_size, data_folder, wav_lst, N_snt, window_len, label_dict, fact_amp
):

    # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    signal_batch = np.zeros([batch_size, window_len])
    label_batch = np.zeros(batch_size)

    snt_id_arr = np.random.randint(N_snt, size=batch_size)

    rand_amp_arr = np.random.uniform(1.0 - fact_amp, 1 + fact_amp, batch_size)

    for i in range(batch_size):

        # select a random sentence from the list
        # [fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst[snt_id_arr[i]])
        # signal=signal.astype(float)/32768
        [signal, fs] = sf.read(op.join(data_folder, wav_lst[snt_id_arr[i]]))

        # accesing to a random chunk
        snt_len = signal.shape[0]
        snt_beg = np.random.randint(snt_len - window_len - 1)  # randint(0, snt_len-2*wlen-1)
        snt_end = snt_beg + window_len

        channels = len(signal.shape)
        if channels == 2:
            print("WARNING: stereo to mono: " + data_folder + wav_lst[snt_id_arr[i]])
            signal = signal[:, 0]

        signal_batch[i, :] = signal[snt_beg:snt_end] * rand_amp_arr[i]
        label_batch[i] = label_dict[wav_lst[snt_id_arr[i]]]

    inputs = Variable(torch.from_numpy(signal_batch).float().cuda().contiguous())
    labels = Variable(torch.from_numpy(label_batch).float().cuda().contiguous())

    return inputs, labels


if __name__ == "__main__":

    # Reading config file
    with open("sincnet_models/sincnet_timit/sincnet_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # [data]
    data_config = config["data"]
    train_data_list = ReadList(data_config["train_files"])
    len_train = len(train_data_list)
    test_data_list = ReadList(data_config["test_files"])
    len_test = len(test_data_list)\
    # Load  label dictionary
    label_dict = np.load(data_config["label_dict"], allow_pickle=True).item()
    # Create output folder
    os.makedirs(data_config["output_folder"], exist_ok=True)

    # [cnn]
    CNN_config = config["CNN"]
    window_len = int(CNN_config["fs"] * CNN_config["convolution_window_len"] / 1000.0)
    window_shift = int(CNN_config["fs"] * CNN_config["convolution_window_shift"] / 1000.0)
    CNN_config.update({"input_dim": window_len})
    CNN_net = SincNet(CNN_config)
    CNN_net.cuda()

    # [dnn]
    DNN1_config = config["DNN_1"]
    DNN1_config.update({"input_dim": CNN_net.out_dim})
    DNN2_config = config["DNN_2"]
    DNN2_config.update({"input_dim": DNN1_config["fc_lay"][-1]})
    DNN1_net = MLP(DNN1_config)
    DNN1_net.cuda()
    DNN2_net = MLP(DNN2_config)
    DNN2_net.cuda()
    
    # [optimization]
    optim_config = config["optimizer"]
    cost_function = nn.NLLLoss()

    # Setting seed for reproducibility
    torch.manual_seed(optim_config["seed"])
    np.random.seed(optim_config["seed"])


    # Set up optimizer
    optimizer_CNN = optim.RMSprop(CNN_net.parameters(), lr=optim_config["lr"], alpha=0.95, eps=1e-8)
    optimizer_DNN1 = optim.RMSprop(DNN1_net.parameters(), lr=optim_config["lr"], alpha=0.95, eps=1e-8)
    optimizer_DNN2 = optim.RMSprop(DNN2_net.parameters(), lr=optim_config["lr"], alpha=0.95, eps=1e-8)
    
    # Training
    for epoch in range(optim_config["n_epochs"]):
        test_flag = 0
        CNN_net.train()
        DNN1_net.train()
        DNN2_net.train()

        loss_sum = 0
        err_sum = 0

        for i in range(optim_config["n_batches"]):
            [inputs, labels] = create_batches_rnd(
                optim_config["batch_size"],
                data_config["data_folder"],
                train_data_list,
                len_train,
                window_len,
                label_dict,
                0.2,
            )

            # Forward pass
            CNN_out = CNN_net(inputs)
            DNN1_out = DNN1_net(CNN_out)
            DNN2_out = DNN2_net(DNN1_out)

            prediction = torch.max(DNN2_out, 1)[1]
            loss = cost_function(DNN2_out, labels.long())
            err = torch.mean((prediction != labels.long()).float())

            # Backward pass
            optimizer_CNN.zero_grad()
            optimizer_DNN1.zero_grad()
            optimizer_DNN2.zero_grad()
            loss.backward()

            # Update weights
            optimizer_CNN.step()
            optimizer_DNN1.step()
            optimizer_DNN2.step()

            loss_sum = loss_sum + loss.detach()
            err_sum = err_sum + err.detach()
        
        loss_total = loss_sum / optim_config["n_batches"]
        err_total = err_sum / optim_config["n_batches"]

        print("Training: Epoch %d, Loss: %.4f, Error: %.4f" % (epoch, loss_total, err_total))

        # Evaluation
        if epoch % optim_config["n_eval_epoch"] == 0:
            CNN_net.eval()
            DNN1_net.eval()
            DNN2_net.eval()

            test_flag = 1
            loss_sum = 0
            err_sum = 0
            err_sum_snt = 0

            with torch.no_grad():
                for i in range(len_test):
                    signal, fs = sf.read(op.join(data_config["data_folder"], test_data_list[i]))

                    signal = torch.from_numpy(signal).float().cuda().contiguous()
                    label_batch = label_dict[test_data_list[i]]

                    # Split signals into chunks
                    begin_sample = 0
                    end_sample = window_len

                    n_fr = int((signal.shape[0] - window_len) / window_shift)

                    signal_array = torch.zeros(optim_config["batch_dev"], window_len).float().cuda().contiguous()
                    label = Variable(
                        (torch.zeros(n_fr + 1) + label_batch).cuda().contiguous().long()
                    )
                    output = Variable(
                        torch.zeros(n_fr + 1, DNN2_config["fc_lay"][-1]).float().cuda().contiguous()
                    )
                    count_fr = 0
                    count_fr_total = 0

                    while end_sample < signal.shape[0]:
                        signal_array[count_fr, :] = signal[begin_sample:end_sample]
                        begin_sample = begin_sample + window_shift
                        end_sample = begin_sample + window_len
                        count_fr += 1
                        count_fr_total += 1

                        if count_fr == optim_config["batch_dev"]:
                            input = Variable(signal_array)
                            
                            CNN_out = CNN_net(input)
                            DNN1_out = DNN1_net(CNN_out)
                            DNN2_out = DNN2_net(DNN1_out)

                            output[count_fr_total - optim_config["batch_dev"] : count_fr_total, :] = DNN2_out
                            
                            count_fr = 0
                            signal_array = torch.zeros(optim_config["batch_dev"], window_len).float().cuda().contiguous()
                    
                    if count_fr > 0:
                        input = Variable(signal_array[:count_fr])
                        CNN_out = CNN_net(input)
                        DNN1_out = DNN1_net(CNN_out)
                        DNN2_out = DNN2_net(DNN1_out)

                        output[count_fr_total - count_fr : count_fr_total, :] = DNN2_out
                    
                    # compute the error rate
                    prediction = torch.max(output, 1)[1]
                    loss = cost_function(output, label.long())
                    err = torch.mean((prediction != label.long()).float())

                    val, best_class = torch.max(torch.sum(output, dim=0), 0)
                    err_sum_snt += (best_class != label[0]).float()

                    loss_sum += loss.detach()
                    err_sum += err.detach()
                
                err_total_dev_snt = err_sum_snt / len_test
                loss_total_dev = loss_sum / len_test
                err_total_dev = err_sum / len_test
            
            print(
                "Validation: Epoch %d, Loss: %.4f, Error: %.4f, Error_snt: %.4f"
                % (epoch, loss_total_dev, err_total_dev, err_total_dev_snt)
            )

            with open(op.join(data_config["output_folder"], "result.res"), "a") as f:
                f.write(
                    "Epoch %d, Loss: %.4f, Error: %.4f, Error_snt: %.4f\n"
                    % (epoch, loss_total_dev, err_total_dev, err_total_dev_snt)
                )
            
            # Save model
            torch.save(CNN_net.state_dict(), op.join(data_config["output_folder"], "cnn_state_dict.pth"))
            torch.save(DNN1_net.state_dict(), op.join(data_config["output_folder"], "dnn1_state_dict.pth"))
            torch.save(DNN2_net.state_dict(), op.join(data_config["output_folder"], "dnn2_state_dict.pth"))