import sys
import os.path as op
import torch.nn as nn

sys.path.append(op.dirname(__file__))
from .SincNet import SincNet, MLP

class SincNetBN(nn.Module):
    def __init__(self, config):
        super(SincNetBN, self).__init__()
        # Converting context and shift in samples
        cnn_config = config.get("CNN", None)
        dnn1_config = config.get("DNN_1", None)
        if cnn_config is None or dnn1_config is None:
            raise ValueError("Config file must contain CNN and DNN_1 keys")
        
        window_len = int(cnn_config["fs"] * cnn_config["convolution_window_len"] / 1000.0)
        cnn_config.update({"input_dim": window_len})
        self.sincnet = SincNet(cnn_config)

        dnn1_config = config["DNN_1"]
        dnn1_config.update({"input_dim": self.sincnet.out_dim})
        self.dnn = MLP(dnn1_config)
        self.out_dim = dnn1_config["fc_lay"][-1]
        
    def forward(self, x):
        x = self.sincnet(x)
        x = self.dnn(x)
        return x