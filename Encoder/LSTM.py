import torch
import torch.nn as nn

class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_space_size, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, latent_space_size)
        self.relu1 = nn.ReLU()

    def forward(self, input_seq):
        lstm_output, (hidden, cell) = self.lstm(input_seq)
        encoder_output = self.fc1(lstm_output)
        encoder_output = self.relu1(encoder_output)
        return encoder_output


# class DecoderLSTM(nn.Module):
#     def __init__(self, hidden_dim, output_dim, num_layers=1):
#         super(DecoderLSTM, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, decoder_input, hidden, cell):
#         decoder_output, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))
#         decoder_output = self.fc(decoder_output)
#         return decoder_output, hidden, cell
