"""Implementation of a Transformer encoder-decoder architecture
    with autoregressive inference, and teacher forcing at training time.
    (as in attention is all you need, but initial input points are unordered)
"""
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointTransformer(nn.Module):
    def __init__(self,
                 d_model=64,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=256,
                 max_seq_len=100,
                 input_dim=3,
                 outdim=6,
                 weight_orient=1.):
        super(PointTransformer, self).__init__()

        self.outdim = outdim
        self.max_seq_len = max_seq_len

        # Embedding layer for (x, y, z) points
        self.segments_embedding = nn.Linear(input_dim, d_model)
        self.points_embedding = nn.Linear(outdim, d_model)

        # Transformer encoder layers (to process unordered input points)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Sinusoidal positional encoding for generated points in the decoder
        self.positional_encoding = self.create_sinusoidal_positional_encoding(max_seq_len, d_model)

        # Transformer decoder layers (to process ordered output sequence)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output layers to predict the next point and the EOS probability
        self.output_layer = nn.Linear(d_model, self.outdim)
        self.eos_layer = nn.Linear(d_model, 1)

    def create_sinusoidal_positional_encoding(self, max_seq_len, d_model):
        """
        Create a tensor of sinusoidal positional encodings.
        
        Args:
        - max_seq_len: The maximum sequence length.
        - d_model: The dimensionality of the embeddings.
        
        Returns:
        - A tensor of shape (max_seq_len, d_model) containing the positional encodings.
        """
        pos = torch.arange(0, max_seq_len).unsqueeze(1)
        i = torch.arange(0, d_model, 2).float()

        angle_rates = 1 / (10000 ** (i / d_model))
        angle_rads = pos * angle_rates

        # Apply sine to even indices and cosine to odd indices
        pos_encoding = torch.zeros(max_seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(angle_rads)
        pos_encoding[:, 1::2] = torch.cos(angle_rads)

        return pos_encoding.unsqueeze(0)  # (1, max_seq_len, d_model)

    def forward(self, src_points, tgt_points=None, src_mask=None):
        src_emb = self.segments_embedding(src_points)
        memory = self.encoder(src_emb.permute(1, 0, 2), src_key_padding_mask=src_mask)

        if tgt_points is not None:
            # Training with provided target points
            sos_point = torch.zeros(tgt_points.size(0), 1, self.outdim).to(tgt_points.device)
            tgt_points = torch.cat((sos_point, tgt_points), dim=1)

            # Embedding and positional encoding
            tgt_emb = self.points_embedding(tgt_points)
            tgt_emb = tgt_emb + self.positional_encoding[:, :tgt_emb.size(1), :].to(tgt_emb.device)
            tgt_len = tgt_emb.size(1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt_emb.device)

            out = self.decoder(tgt_emb.permute(1, 0, 2), memory, tgt_mask=tgt_mask)
            output_points = self.output_layer(out.permute(1, 0, 2))

            # todo: config.weight_orient on normals.


            eos_logits = self.eos_layer(out.permute(1, 0, 2))
            eos_probs = torch.sigmoid(eos_logits)  # Convert logits to probabilities
            return output_points, eos_probs
        else:
            # Inference: Autoregressive generation starting with (0, 0, 0)
            output_points = []
            eos_probs = []

            # Initialize with (0, 0, 0) point as the SOS token
            sos_token = torch.zeros(1, 1, self.outdim).to(src_points.device)
            tgt_emb = self.points_embedding(sos_token)
            tgt_emb = tgt_emb + self.positional_encoding[:, :1, :].to(tgt_emb.device)  # Add positional encoding only once for SOS

            for i in range(self.max_seq_len):  # Max number of points
                out = self.decoder(tgt_emb.permute(1, 0, 2), memory)
                next_point = self.output_layer(out[-1])
                eos_logit = self.eos_layer(out[-1])
                eos_prob = torch.sigmoid(eos_logit)

                output_points.append(next_point.unsqueeze(1))
                eos_probs.append(eos_prob.unsqueeze(1))

                if eos_prob.item() > 0.5:
                    break

                # Embed the next point and add positional encoding for the next position
                next_emb = self.points_embedding(next_point.unsqueeze(1))
                next_emb = next_emb + self.positional_encoding[:, i+1:i+2, :].to(next_emb.device)
                tgt_emb = torch.cat((tgt_emb, next_emb), dim=1)

            print(f'Stop after {i} points.')

            output_points = torch.cat(output_points, dim=1)
            eos_probs = torch.cat(eos_probs, dim=1)
            return output_points, eos_probs

    def generate_square_subsequent_mask(self, sz):
        """
        Generate a causal mask to prevent the decoder from attending to future tokens.
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask