import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, input):
        pass


class TransformerVAE(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        latent_dim,
        n_head,
        n_encoder_layers,
        n_decoder_layers,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_head, dim_feedforward=512, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim, nhead=n_head, dim_feedforward=512, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        # self.mem2mean = nn.Linear()
        self.output2logits = nn.Linear(embedding_dim, vocab_size)


class RVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        latent_size: int,
        with_att: bool,
        device: str,
        num_layers=1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_size, padding_idx=0, max_norm=1
        )
        self.encoder = EncoderRNN(
            self.embedding, embedding_size, hidden_size, device, num_layers=num_layers
        )
        if with_att:
            self.decoder = AttnDecoderRNN(
                self.embedding,
                embedding_size,
                hidden_size,
                vocab_size,
                device,
                num_layers=num_layers,
            )
        else:
            self.decoder = DecoderRNN(
                self.embedding,
                embedding_size,
                hidden_size,
                vocab_size,
                device,
                num_layers=num_layers,
            )
        self.hidden2mu = nn.Linear(hidden_size, latent_size)
        self.hidden2logvar = nn.Linear(hidden_size, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size)
        self.device = device

    def forward(self, encoder_inputs):
        # encode
        encoder_outputs, hidden = self.encoder(encoder_inputs)
        mu, logvar = self.hidden2mu(hidden), self.hidden2logvar(hidden)
        # latent reparam
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)
        # decode
        hidden = self.latent2hidden(z)
        decoder_outputs, _, _ = self.decoder(encoder_outputs, hidden, encoder_inputs)
        # ouptuts are logits
        return decoder_outputs, mu, logvar


class EncoderRNN(nn.Module):
    def __init__(
        self,
        embedding,
        embedding_size: int,
        hidden_size: int,
        device,
        num_layers=1,
        dropout_p=0.1,
    ):
        super().__init__()
        self.device = device

        self.embedding = embedding
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(
        self,
        embedding,
        embedding_size: int,
        hidden_size: int,
        output_size: int,
        device,
        num_layers=1,
    ):
        super().__init__()
        self.device = device

        self.embedding = embedding
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        max_length = encoder_outputs.size(1)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=self.device
        ).fill_(BOS_IDX)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(max_length):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden
            )
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(
                    -1
                ).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return (
            decoder_outputs,
            decoder_hidden,
            None,
        )  # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(
        self,
        embedding,
        embedding_size: int,
        hidden_size: int,
        output_size: int,
        device: str,
        num_layers=1,
        dropout_p=0.1,
    ):
        super().__init__()
        self.device = device

        self.embedding = embedding
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(
            embedding_size + hidden_size, hidden_size, num_layers, batch_first=True
        )
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        max_length = encoder_outputs.size(1)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=self.device
        ).fill_(BOS_IDX)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(max_length):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(
                    -1
                ).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
