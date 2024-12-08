import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DoubleLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, dropout, device, num_layers=2, bidirectional=False):
        super(DoubleLSTMClassifier, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Embeddings
        self.tcr_embedding = nn.Embedding(20 + 1, embedding_dim, padding_idx=0)
        self.pep_embedding = nn.Embedding(20 + 1, embedding_dim, padding_idx=0)

        # LSTMs
        self.tcr_lstm = nn.LSTM(embedding_dim, lstm_dim, 
                                num_layers=self.num_layers, 
                                batch_first=True, 
                                dropout=dropout, 
                                bidirectional=self.bidirectional)
        self.pep_lstm = nn.LSTM(embedding_dim, lstm_dim, 
                                num_layers=self.num_layers, 
                                batch_first=True, 
                                dropout=dropout, 
                                bidirectional=self.bidirectional)

        # Because we have two sequences (tcr and pep) and each LSTM may be bidirectional,
        # the final concatenated dimension = self.num_directions * lstm_dim for TCR 
        # plus self.num_directions * lstm_dim for PEP = 2 * self.num_directions * lstm_dim
        mlp_input_dim = 2 * self.num_directions * lstm_dim

        self.hidden_layer = nn.Linear(mlp_input_dim, lstm_dim)
        self.relu = torch.nn.LeakyReLU()
        self.output_layer = nn.Linear(lstm_dim, 1)
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.lstm_dim)).to(self.device),
                autograd.Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.lstm_dim)).to(self.device))
    
    def lstm_pass(self, lstm, padded_embeds, lengths):
        # Sort by length (descending)
        lengths, perm_idx = lengths.sort(0, descending=True)
        padded_embeds = padded_embeds[perm_idx]
        # Pack the batch and ignore the padding
        # packed_input = pack_padded_sequence(padded_embeds, lengths, batch_first=True)
        padded_embeds = torch.nn.utils.rnn.pack_padded_sequence(padded_embeds, lengths.cpu().to(torch.int64), batch_first=True)
        # Initialize hidden
        hidden = self.init_hidden(len(lengths))
        # Feed into LSTM
        lstm_out, hidden = lstm(padded_embeds, hidden)
        # Unpack
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        # Revert sorting
        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]
        return lstm_out
    # The rest of the code remains the same, except when taking the last cell from the LSTM output.
    # We need to account for bidirection. For LSTM outputs, if bidirectional=True, the last timestep
    # has [forward_out; backward_out]. We pick the last timestep accordingly.
    def forward(self, tcrs, tcr_lens, peps, pep_lens):
        # TCR Encoding
        tcr_embeds = self.tcr_embedding(tcrs)
        tcr_lstm_out = self.lstm_pass(self.tcr_lstm, tcr_embeds, tcr_lens)
        # For bidirectional: last cell of forward is tcr_lstm_out[i, tcr_lens[i]-1, :lstm_dim]
        # and last cell of backward is tcr_lstm_out[i, 0, lstm_dim:]
        tcr_last_cell = []
        for i, length in enumerate(tcr_lens):
            length = length.item()
            if self.bidirectional:
                forward_vec = tcr_lstm_out[i, length - 1, :self.lstm_dim]
                backward_vec = tcr_lstm_out[i, 0, self.lstm_dim:]
                tcr_last_cell.append(torch.cat([forward_vec, backward_vec], dim=0))
            else:
                tcr_last_cell.append(tcr_lstm_out[i, length - 1, :self.lstm_dim])
        tcr_last_cell = torch.stack(tcr_last_cell)

        # PEPTIDE Encoding
        pep_embeds = self.pep_embedding(peps)
        pep_lstm_out = self.lstm_pass(self.pep_lstm, pep_embeds, pep_lens)
        pep_last_cell = []
        for i, length in enumerate(pep_lens):
            length = length.item()
            if self.bidirectional:
                forward_vec = pep_lstm_out[i, length - 1, :self.lstm_dim]
                backward_vec = pep_lstm_out[i, 0, self.lstm_dim:]
                pep_last_cell.append(torch.cat([forward_vec, backward_vec], dim=0))
            else:
                pep_last_cell.append(pep_lstm_out[i, length - 1, :self.lstm_dim])
        pep_last_cell = torch.stack(pep_last_cell)

        # MLP Classifier
        tcr_pep_concat = torch.cat([tcr_last_cell, pep_last_cell], 1)
        hidden_output = self.dropout(self.relu(self.hidden_layer(tcr_pep_concat)))
        mlp_output = self.output_layer(hidden_output)
        output = torch.sigmoid(mlp_output)
        return output


class PaddingAutoencoder(nn.Module):
    def __init__(self, input_len, input_dim, encoding_dim):
        super(PaddingAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.input_len = input_len
        self.encoding_dim = encoding_dim
        # Encoder
        self.encoder = nn.Sequential(
                    nn.Linear(self.input_len * self.input_dim, 300),
                    nn.ELU(),
                    nn.Dropout(0.1),
                    nn.Linear(300, 100),
                    nn.ELU(),
                    nn.Dropout(0.1),
                    nn.Linear(100, self.encoding_dim))
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.encoding_dim, 100),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(100, 300),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(300, self.input_len * self.input_dim))

    def forward(self, batch_size, padded_input):
        concat = padded_input.view(batch_size, self.input_len * self.input_dim)
        encoded = self.encoder(concat)
        decoded = self.decoder(encoded)
        decoding = decoded.view(batch_size, self.input_len, self.input_dim)
        decoding = F.softmax(decoding, dim=2)
        return decoding
    pass


class AutoencoderLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, device, max_len, input_dim, encoding_dim, batch_size, ae_file, train_ae):
        super(AutoencoderLSTMClassifier, self).__init__()
        # GPU
        self.device = device
        # Dimensions
        self.embedding_dim = embedding_dim
        self.lstm_dim = encoding_dim
        self.max_len = max_len
        self.input_dim = input_dim
        self.batch_size = batch_size
        # TCR Autoencoder
        self.autoencoder = PaddingAutoencoder(max_len, input_dim, encoding_dim)
        checkpoint = torch.load(ae_file, map_location=device)
        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        if train_ae is False:
            for param in self.autoencoder.parameters():
                param.requires_grad = False
        self.autoencoder.eval()
        # Embedding matrices - 20 amino acids + padding
        self.pep_embedding = nn.Embedding(20 + 1, embedding_dim, padding_idx=0)
        # RNN - LSTM
        self.pep_lstm = nn.LSTM(embedding_dim, self.lstm_dim, num_layers=2, batch_first=True, dropout=0.1)
        # MLP
        self.mlp_dim = self.lstm_dim + encoding_dim
        self.hidden_layer = nn.Linear(self.mlp_dim, self.mlp_dim // 2)
        self.relu = torch.nn.LeakyReLU()
        self.output_layer = nn.Linear(self.mlp_dim // 2, 1)
        self.dropout = nn.Dropout(p=0.1)

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)).to(self.device),
                autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)).to(self.device))

    def lstm_pass(self, lstm, padded_embeds, lengths):
        # Before using PyTorch pack_padded_sequence we need to order the sequences batch by descending sequence length
        lengths, perm_idx = lengths.sort(0, descending=True)
        padded_embeds = padded_embeds[perm_idx]
        # Pack the batch and ignore the padding
        # padded_embeds = torch.nn.utils.rnn.pack_padded_sequence(padded_embeds, lengths, batch_first=True)
        padded_embeds = torch.nn.utils.rnn.pack_padded_sequence(padded_embeds, lengths.cpu().to(torch.int64), batch_first=True)
        # Initialize the hidden state
        batch_size = len(lengths)
        hidden = self.init_hidden(batch_size)
        # Feed into the RNN
        lstm_out, hidden = lstm(padded_embeds, hidden)
        # Unpack the batch after the RNN
        lstm_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # Remember that our outputs are sorted. We want the original ordering
        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]
        lengths = lengths[unperm_idx]
        return lstm_out

    def forward(self, padded_tcrs, peps, pep_lens):
        # TCR Encoder:
        # Embedding
        concat = padded_tcrs.view(self.batch_size, self.max_len * self.input_dim)
        encoded_tcrs = self.autoencoder.encoder(concat)
        # PEPTIDE Encoder:
        # Embedding
        pep_embeds = self.pep_embedding(peps)
        # LSTM Acceptor
        pep_lstm_out = self.lstm_pass(self.pep_lstm, pep_embeds, pep_lens)
        pep_last_cell = torch.cat([pep_lstm_out[i, j.data - 1] for i, j in enumerate(pep_lens)]).view(len(pep_lens), self.lstm_dim)
        # MLP Classifier
        tcr_pep_concat = torch.cat([encoded_tcrs, pep_last_cell], 1)
        hidden_output = self.dropout(self.relu(self.hidden_layer(tcr_pep_concat)))
        mlp_output = self.output_layer(hidden_output)
        output = F.sigmoid(mlp_output)
        return output
