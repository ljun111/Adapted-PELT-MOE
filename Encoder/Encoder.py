import torch
import torch.nn as nn
import torch.nn.init as init

class Encoder_base(nn.Module):
    def __init__(self, input_size, latent_space_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear1 = nn.Linear(input_size, input_size // 2)
        # self.relu1 = nn.ReLU()
        # self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        # self.linear2 = nn.Linear(input_size // 2, input_size // 4)
        self.relu2 = nn.ReLU()
        # self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.linear3 = nn.Linear(input_size // 2, latent_space_size)

    def forward(self, x):
        x = self.linear1(x)
        # x = self.relu1(x)
        # x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

class Encoder_base_2(nn.Module):
    def __init__(self, input_size, latent_space_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear1 = nn.Linear(input_size, latent_space_size)
        # self.relu1 = nn.ReLU()
        # self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        # self.linear2 = nn.Linear(input_size // 2, input_size // 4)
        self.relu2 = nn.ReLU()
        # self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        # self.linear3 = nn.Linear(input_size // 2, latent_space_size)

    def forward(self, x):
        x = self.linear1(x)
        # x = self.relu1(x)
        # x = self.linear2(x)
        x = self.relu2(x)
        # x = self.linear3(x)
        return x

class Encoder_base_3(nn.Module):
    def __init__(self, input_size, latent_space_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear1 = nn.Linear(input_size, input_size // 2)
        self.relu1 = nn.ReLU()
        # self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.linear2 = nn.Linear(input_size // 2, input_size // 4)
        self.relu2 = nn.ReLU()
        # self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.linear3 = nn.Linear(input_size // 4, latent_space_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

class Encoder_base_4(nn.Module):
    def __init__(self, input_size, latent_space_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear1 = nn.Linear(input_size, input_size // 2)
        # self.relu1 = nn.ReLU()
        # self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        # self.linear2 = nn.Linear(input_size // 2, input_size // 4)
        self.relu2 = nn.ReLU()
        # self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.linear3 = nn.Linear(input_size // 2, latent_space_size // 2)

    def forward(self, x):
        x = self.linear1(x)
        # x = self.relu1(x)
        # x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

class Encoder_base_5(nn.Module):
    def __init__(self, input_size, latent_space_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear1 = nn.Linear(input_size, input_size // 2)
        # self.relu1 = nn.ReLU()
        # self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        # self.linear2 = nn.Linear(input_size // 2, input_size // 4)
        self.relu2 = nn.ReLU()
        # self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.linear3 = nn.Linear(input_size // 2, latent_space_size // 4)

    def forward(self, x):
        x = self.linear1(x)
        # x = self.relu1(x)
        # x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

class Decoder_base(nn.Module):
    def __init__(self, input_size, latent_space_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear1 = nn.Linear(latent_space_size, input_size // 2)
        # self.relu1 = nn.ReLU()
        # self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        # self.linear2 = nn.Linear(input_size // 4, input_size // 2)
        # self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(input_size // 2, input_size)


    def forward(self, x, last_layer=False):
        x = self.linear1(x)
        # x = self.relu1(x)
        # x = self.linear2(x)
        # x = self.relu2(x)
        out = self.linear3(x)
        if last_layer is False:
            return out
        else:
            return x

