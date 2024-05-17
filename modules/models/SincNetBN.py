import sys
import os.path as op
import torch.nn as nn

sys.path.append(op.dirname(__file__))
from SincNet import SincNet, MLP

class SincNetBN(nn.Module):
    def __init__(self, config):
        super(SincNetBN, self).__init__()
        # Converting context and shift in samples
        cnn_config = config.get("cnn", None)
        dnn_config = config.get("dnn", None)
        if cnn_config is None or dnn_config is None:
            raise ValueError("Config file must contain `cnn` and `dnn` keys")
        
        window_len = int(cnn_config["fs"] * cnn_config["convolution_window_len"] / 1000.0)
        cnn_config.update({"input_dim": window_len})
        self.sincnet = SincNet(cnn_config)

        dnn_config = config["dnn"]
        dnn_config.update({"input_dim": self.sincnet.out_dim})
        self.dnn = MLP(dnn_config)
        self.out_dim = dnn_config["fc_lay"][-1]
        
    def forward(self, x):
        x = self.sincnet(x)
        x = self.dnn(x)
        return x