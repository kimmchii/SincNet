from dnn_models import SincNet, MLP
import torch
import yaml

avoid_small_en_fr = True
energy_th = 0.1  # Avoid frames with an energy that is 1/10 over the average energy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = None

if __name__ == "__main__":
    # Load model config
    with open("sincnet_models/SincNet_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Converting context and shift in samples
    CNN_config = config["CNN"]
    window_len = int(CNN_config["fs"] * CNN_config["convolution_window_len"] / 1000.0)
    CNN_config.update({"input_dim": window_len})
    CNN_net = SincNet(CNN_config)

    DNN1_config = config["DNN_1"]
    DNN1_config.update({"input_dim": CNN_net.out_dim})
    DNN1_net = MLP(DNN1_config)

    DNN2_config = config["DNN_2"]
    DNN2_config.update({"input_dim": DNN1_config["fc_lay"][-1]})
    DNN2_net = MLP(DNN2_config)

    # Load model state dicts
    cnn_state_dict = torch.load("sincnet_models/SincNet_TIMIT/train/cnn_state_dict.pth")
    dnn1_state_dict = torch.load("sincnet_models/SincNet_TIMIT/train/dnn1_state_dict.pth")
    dnn2_state_dict = torch.load("sincnet_models/SincNet_TIMIT/train/dnn2_state_dict.pth")

    CNN_net.load_state_dict(cnn_state_dict)
    DNN1_net.load_state_dict(dnn1_state_dict)
    DNN2_net.load_state_dict(dnn2_state_dict)


