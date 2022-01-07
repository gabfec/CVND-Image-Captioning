import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

        self.hidden_size = hidden_size

    def forward(self, features, captions):
        # remove the <end> from captions
        embeddings = self.embed(captions[:, :-1])

        # Concatenate the features and captions
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hidden, states = self.lstm(inputs)
        outputs = self.linear(hidden)

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "

        states = (torch.randn(1, 1, self.hidden_size).to(inputs.device),
                  torch.randn(1, 1, self.hidden_size).to(inputs.device))

        sentence = []
        for i in range(max_len):
            hidden, states = self.lstm(inputs, states)
            scores = self.linear(hidden)
            predicted = torch.argmax(scores, dim=2)
            sentence.append(predicted.item())
            if predicted.item() == 1:
                break
            inputs = self.embed(predicted)

        return sentence
