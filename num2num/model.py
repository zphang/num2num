import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .data import NOTHING_IDX


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.num_layers = config.num_layers
        self.rnn_size = config.rnn_size
        self.bidirectional_encoder = config.bidirectional_encoder

        self.embedding = nn.Embedding(
            num_embeddings=config.input_vocab_size,
            embedding_dim=self.rnn_size,
            padding_idx=NOTHING_IDX,
        )

        self.rnn = nn.LSTM(
            input_size=self.rnn_size,
            hidden_size=(
                self.rnn_size // 2
                if self.bidirectional_encoder
                else self.rnn_size),
            num_layers=self.num_layers,
            bidirectional=self.bidirectional_encoder,
        )

    def forward(self, x, x_len, hidden):
        x = x.permute(1, 0)
        embedded = self.embedding(x)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, x_len,
        )
        lstm_out, hidden = self.rnn(packed_embedded, hidden)

        lstm_out = nn.utils.rnn.pad_packed_sequence(lstm_out)

        if self.bidirectional_encoder:
            new_hidden = []
            for i in range(2):
                temp_new_hidden = hidden[i] \
                    .view(self.num_layers, 2, *hidden[i].size()[1:]) \
                    .transpose(1, 3).transpose(1, 2).contiguous() \
                    .view(self.num_layers, hidden[i].size(1),
                          hidden[i].size(2) * 2)

                new_hidden.append(temp_new_hidden)
            hidden = tuple(new_hidden)

        return (
            lstm_out[0],
            hidden,
        )


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.num_layers = config.num_layers
        self.rnn_size = config.rnn_size

        self.embedding = nn.Embedding(
            num_embeddings=config.output_vocab_size,
            embedding_dim=self.rnn_size,
        )
        self.rnn = nn.LSTM(
            self.rnn_size,
            self.rnn_size,
            self.num_layers,
        )
        self.fc1 = nn.Linear(self.rnn_size, config.output_vocab_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, x, hidden, full_encoder_ouput):
        x = x.permute(1, 0)
        embedded = self.embedding(x)
        lstm_out, hidden = self.rnn(embedded, hidden)
        output = self.fc1(
            lstm_out.view(lstm_out.size(0) * lstm_out.size(1), lstm_out.size(2))
        ).view(lstm_out.size(0), lstm_out.size(1), self.fc1.out_features)
        log_softmax_output = \
            F.log_softmax(output.permute(2, 1, 0)).permute(2, 1, 0)
        return (
            log_softmax_output.permute(1, 0, 2),
            hidden,
            dict(),
        )

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        h = Variable(
            weight.new(self.num_layers, batch_size, self.rnn_size).zero_()
        )
        c = Variable(
            weight.new(self.num_layers, batch_size, self.rnn_size).zero_()
        )
        return h, c


class Attn(nn.Module):
    def __init__(self, config):
        super(Attn, self).__init__()

        self.method = config.attn_method
        self.hidden_size = config.rnn_size
        self.cuda = config.cuda

        if self.method == 'dot':
            pass
        elif self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            hidden_size_mult = 2
            self.attn = nn.Linear(
                self.hidden_size * hidden_size_mult,
                self.hidden_size
            )
            self.v = nn.Linear(self.hidden_size, 1)
        else:
            raise KeyError("Unknown method")

    def forward(self, hidden, encoder_outputs):
        scores = self.score(hidden, encoder_outputs)  # I*B

        return F.softmax(scores.t()).t()

    def score(self, hidden, encoder_outputs):
        """
        hidden: [(batch_size) * (hidden_dim)]
        encoder_output: [(input_length) * (batch_size) * (hidden_dim)]

        Returns: [(input_length) * (batch_size)]
        """

        if self.method == 'dot':
            return torch.bmm(
                hidden.unsqueeze(1),  # B*H
                encoder_outputs.permute(1, 2, 0),  # B*H*I
            ).squeeze(1).permute(1, 0)

        elif self.method == 'general':
            return torch.bmm(
                hidden.unsqueeze(1),  # B*H
                self.attn(encoder_outputs.permute(1, 2, 0)),  # B*H*I
            ).squeeze(1).permute(1, 0)

        elif self.method == 'concat':
            energy = self.attn(torch.cat((
                hidden.expand(encoder_outputs.size()[0], *hidden.size()),
                encoder_outputs,
            ), 2).permute(1, 0, 2))  # B*I*H
            return self.v(torch.tanh(energy)).squeeze(2).permute(1, 0)
        else:
            raise KeyError("Unknown method")


class AttnDecoder(nn.Module):
    def __init__(self, config):
        super(AttnDecoder, self).__init__()
        self.num_layers = config.num_layers
        self.rnn_size = config.rnn_size

        self.embedding = nn.Embedding(
            num_embeddings=config.output_vocab_size,
            embedding_dim=self.rnn_size,
        )
        self.attn = Attn(config)
        self.rnn = nn.LSTM(
            self.rnn_size,
            self.rnn_size,
            self.num_layers,
        )
        self.fc1 = nn.Linear(self.rnn_size * 2, config.output_vocab_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, x, hidden, full_encoder_output):
        x = x.permute(1, 0)
        embedded = self.embedding(x)

        lstm_out, hidden = self.rnn(embedded, hidden)

        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(hidden[0][-1], full_encoder_output)
        context = torch.bmm(
            attn_weights.t().unsqueeze(1),
            full_encoder_output.transpose(0, 1),
        ).transpose(0, 1)
        context = context / attn_weights.size()[0]  # 1*B*H

        h_tilde = torch.cat((embedded, context), 2)
        output = self.fc1(h_tilde)
        log_softmax_output = \
            F.log_softmax(output.permute(2, 1, 0)).permute(2, 1, 0)

        return (
            log_softmax_output.permute(1, 0, 2),
            hidden,
            {"attn": attn_weights},
        )

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        h = Variable(
            weight.new(self.num_layers, batch_size, self.rnn_size * 2).zero_()
        )
        c = Variable(
            weight.new(self.num_layers, batch_size, self.rnn_size * 2).zero_()
        )
        return h, c
