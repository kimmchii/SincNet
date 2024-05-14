import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeakerCounter(nn.Module):
    def __init__(self, input_size=1024, attention_dim=128, num_heads=8, num_classes=10):
        super(SpeakerCounter, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, batch_first=True)
        self.fc1 = nn.Linear(input_size, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x, need_weights=False)
        attn_output = attn_output.mean(dim=0)  # Aggregate the attention output
        x = F.relu(self.fc1(attn_output))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    model = SpeakerCounter(input_size=1024, attention_dim=128, num_heads=8, num_classes=10)
    input_tensor = torch.randn(1, 1, 1024)
    output = model(input_tensor)
    print(output.shape)  # should output torch.Size([32, 10])