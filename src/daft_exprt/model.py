import numpy as np
import torch

from collections import namedtuple

from torch import nn
from torch.autograd import Function
from torch.distributions import Normal
from torch.nn.parameter import Parameter

from daft_exprt.extract_features import duration_to_integer


def get_mask_from_lengths(lengths):
    ''' Create a masked tensor from given lengths

    :param lengths:     torch.tensor of size (B, ) -- lengths of each example

    :return mask: torch.tensor of size (B, max_length) -- the masked tensor
    '''
    max_len = torch.max(lengths)
    ids = torch.arange(0, max_len).cuda(lengths.device, non_blocking=True).long()
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    ''' Gradient Reversal Layer
            Y. Ganin, V. Lempitsky,
            "Unsupervised Domain Adaptation by Backpropagation",
            in ICML, 2015.
        Forward pass is the identity function
        In the backward pass, upstream gradients are multiplied by -lambda (i.e. gradient are reversed)
    '''
    def __init__(self, hparams):
        super(GradientReversal, self).__init__()
        self.lambda_ = hparams.lambda_reversal

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class LinearNorm(nn.Module):
    ''' Linear Norm Module:
        - Linear Layer
    '''
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        ''' Forward function of Linear Norm
            x = (*, in_dim)
        '''
        x = self.linear_layer(x)  # (*, out_dim)
        
        return x


class ConvNorm1D(nn.Module):
    ''' Conv Norm 1D Module:
        - Conv 1D
    '''
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))
    
    def forward(self, x):
        ''' Forward function of Conv Norm 1D
            x = (B, L, in_channels)
        '''
        x = x.transpose(1, 2)  # (B, in_channels, L)
        x = self.conv(x)  # (B, out_channels, L)
        x = x.transpose(1, 2)  # (B, L, out_channels)
        
        return x


class ConvNorm2D(nn.Module):
    ''' Conv Norm 2D Module:
        - Conv 2D
    '''
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        ''' Forward function of Conv Norm 2D:
            x = (B, H, W, in_channels)
        '''
        x = x.permute(0, 3, 1, 2)  # (B, in_channels, H, W)
        x = self.conv(x)  # (B, out_channels, H, W)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, out_channels)
        
        return x


class PositionalEncoding(nn.Module):
    ''' Positional Encoding Module:
        - Sinusoidal Positional Embedding
    '''
    def __init__(self, embed_dim, max_len=5000, timestep=10000.):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim 
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-np.log(timestep) / self.embed_dim))  # (embed_dim // 2, )
        self.pos_enc = torch.FloatTensor(max_len, self.embed_dim).zero_()  # (max_len, embed_dim)
        self.pos_enc[:, 0::2] = torch.sin(pos * div_term)
        self.pos_enc[:, 1::2] = torch.cos(pos * div_term)
    
    def forward(self, x):
        ''' Forward function of Positional Encoding:
            x = (B, N) -- Long or Int tensor
        '''
        # initialize tensor
        nb_frames_max = torch.max(torch.cumsum(x, dim=1))
        pos_emb = torch.FloatTensor(x.size(0), nb_frames_max, self.embed_dim).zero_()  # (B, nb_frames_max, embed_dim)
        pos_emb = pos_emb.cuda(x.device, non_blocking=True).float()  # (B, nb_frames_max, embed_dim)
        
        # can be used for absolute or relative positioning
        for line_idx in range(x.size(0)):
            pos_idx = []
            for column_idx in range(x.size(1)):
                idx = x[line_idx, column_idx]
                pos_idx.extend([i for i in range(idx)])
            emb = self.pos_enc[pos_idx]  # (nb_frames, embed_dim)
            pos_emb[line_idx, :emb.size(0), :] = emb
        
        return pos_emb


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention Module:
        - Multi-Head Attention
            A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser and I. Polosukhin
            "Attention is all you need",
            in NeurIPS, 2017.
        - Dropout
        - Residual Connection 
        - Layer Normalization
    '''
    def __init__(self, hparams):
        super(MultiHeadAttention, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(hparams.hidden_embed_dim,
                                                          hparams.attn_nb_heads,
                                                          hparams.attn_dropout)
        self.dropout = nn.Dropout(hparams.attn_dropout)
        self.layer_norm = nn.LayerNorm(hparams.hidden_embed_dim)
    
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        ''' Forward function of Multi-Head Attention:
            query = (B, L_max, hidden_embed_dim)
            key = (B, T_max, hidden_embed_dim)
            value = (B, T_max, hidden_embed_dim)
            key_padding_mask = (B, T_max) if not None
            attn_mask = (L_max, T_max) if not None
        '''
        # compute multi-head attention
        # attn_outputs = (L_max, B, hidden_embed_dim)
        # attn_weights = (B, L_max, T_max)
        attn_outputs, attn_weights = self.multi_head_attention(query.transpose(0, 1),
                                                               key.transpose(0, 1),
                                                               value.transpose(0, 1),
                                                               key_padding_mask=key_padding_mask,
                                                               attn_mask=attn_mask)
        attn_outputs = attn_outputs.transpose(0, 1)  # (B, L_max, hidden_embed_dim)
        # apply dropout
        attn_outputs = self.dropout(attn_outputs)  # (B, L_max, hidden_embed_dim)
        # add residual connection and perform layer normalization
        attn_outputs = self.layer_norm(attn_outputs + query)  # (B, L_max, hidden_embed_dim)

        return attn_outputs, attn_weights


class PositionWiseConvFF(nn.Module):
    ''' Position Wise Convolutional Feed-Forward Module:
        - 2x Conv 1D with ReLU
        - Dropout
        - Residual Connection 
        - Layer Normalization
        - FiLM conditioning (if film_params is not None)
    '''
    def __init__(self, hparams):
        super(PositionWiseConvFF, self).__init__()
        self.convs = nn.Sequential(
            ConvNorm1D(hparams.hidden_embed_dim, hparams.conv_channels,
                       kernel_size=hparams.conv_kernel, stride=1,
                       padding=int((hparams.conv_kernel - 1) / 2),
                       dilation=1, w_init_gain='relu'),
            nn.ReLU(),
            ConvNorm1D(hparams.conv_channels, hparams.hidden_embed_dim,
                       kernel_size=hparams.conv_kernel, stride=1,
                       padding=int((hparams.conv_kernel - 1) / 2),
                       dilation=1, w_init_gain='linear'),
            nn.Dropout(hparams.conv_dropout)
        )
        self.layer_norm = nn.LayerNorm(hparams.hidden_embed_dim)
    
    def forward(self, x, film_params):
        ''' Forward function of PositionWiseConvFF:
            x = (B, L_max, hidden_embed_dim)
            film_params = (B, nb_film_params)
        '''
        # pass through convs
        outputs = self.convs(x)  # (B, L_max, hidden_embed_dim)
        # add residual connection and perform layer normalization
        outputs = self.layer_norm(outputs + x)  # (B, L_max, hidden_embed_dim)
        # add FiLM transformation
        if film_params is not None:
            nb_gammas = int(film_params.size(1) / 2)
            assert(nb_gammas == outputs.size(2))
            gammas = film_params[:, :nb_gammas].unsqueeze(1)  # (B, 1, hidden_embed_dim)
            betas = film_params[:, nb_gammas:].unsqueeze(1)  # (B, 1, hidden_embed_dim)
            outputs = gammas * outputs + betas  # (B, L_max, hidden_embed_dim)
        
        return outputs


class FFTBlock(nn.Module):
    ''' FFT Block Module:
        - Multi-Head Attention
        - Position Wise Convolutional Feed-Forward
        - FiLM conditioning (if film_params is not None)
    '''
    def __init__(self, hparams):
        super(FFTBlock, self).__init__()
        self.attention = MultiHeadAttention(hparams)
        self.feed_forward = PositionWiseConvFF(hparams)
    
    def forward(self, x, film_params, mask):
        ''' Forward function of FFT Block:
            x = (B, L_max, hidden_embed_dim)
            film_params = (B, nb_film_params)
            mask = (B, L_max)
        '''
        # attend
        attn_outputs, _ = self.attention(x, x, x, key_padding_mask=mask)  # (B, L_max, hidden_embed_dim)
        attn_outputs = attn_outputs.masked_fill(mask.unsqueeze(2), 0)  # (B, L_max, hidden_embed_dim)
        # feed-forward pass
        outputs = self.feed_forward(attn_outputs, film_params)  # (B, L_max, hidden_embed_dim)
        outputs = outputs.masked_fill(mask.unsqueeze(2), 0)  # (B, L_max, hidden_embed_dim)
        
        return outputs


class SpeakerClassifier(nn.Module):
    ''' Speaker Classifier Module:
        - 3x Linear Layers with ReLU
    '''
    def __init__(self, hparams):
        super(SpeakerClassifier, self).__init__()
        nb_speakers = hparams.n_speakers - 1
        embed_dim = hparams.prosody_encoder['hidden_embed_dim']
        
        self.classifier = nn.Sequential(
            GradientReversal(hparams),
            LinearNorm(embed_dim, embed_dim, w_init_gain='relu'),
            nn.ReLU(),
            LinearNorm(embed_dim, embed_dim, w_init_gain='relu'),
            nn.ReLU(),
            LinearNorm(embed_dim, nb_speakers, w_init_gain='linear')
        )
    
    def forward(self, x):
        ''' Forward function of Speaker Classifier:
            x = (B, embed_dim)
        '''
        # pass through classifier
        outputs = self.classifier(x)  # (B, nb_speakers)
        
        return outputs


class ProsodyEncoder(nn.Module):
    ''' Prosody Encoder Module:
        - Positional Encoding
        - Energy Embedding:
            - 1x Conv 1D
        - Pitch Embedding:
            - 1x Conv 1D
        - Mel-Spec PreNet:
            - 3x Conv 1D
        - 4x FFT Blocks
        - Speaker Embedding
        - Linear Projection Layer
        
        This module predicts FiLM parameters to condition the Core Acoustic Model
        References:
        - E. Perez, F. Strub, H. de Vries, V. Dumoulin and A. Courville,
          "FiLM: Visual Reasoning with a General Conditioning Layer", in AAAI, 2018.
        - https://ml-retrospectives.github.io/neurips2019/accepted_retrospectives/2019/film/
        - https://distill.pub/2018/feature-wise-transformations/
        - B.N. Oreshkin, P. Rodriguez and A. Lacoste,
          "TADAM: Task dependent adaptive metric for improved few-shot learning", arXiv:1805.10123, 2018.
    '''
    def __init__(self, hparams):
        super(ProsodyEncoder, self).__init__()
        n_speakers = hparams.n_speakers
        nb_mels = hparams.n_mel_channels
        self.post_mult_weight = hparams.post_mult_weight
        self.module_params = {
            'encoder': (hparams.phoneme_encoder['nb_blocks'], hparams.phoneme_encoder['hidden_embed_dim']),
            'prosody_predictor': (hparams.local_prosody_predictor['nb_blocks'], hparams.local_prosody_predictor['conv_channels']),
            'decoder': (hparams.frame_decoder['nb_blocks'], hparams.phoneme_encoder['hidden_embed_dim'])
        }
        Tuple = namedtuple('Tuple', hparams.prosody_encoder)
        hparams = Tuple(**hparams.prosody_encoder)
        
        # positional encoding
        self.pos_enc = PositionalEncoding(hparams.hidden_embed_dim)
        # energy embedding
        self.energy_embedding = ConvNorm1D(1, hparams.hidden_embed_dim, kernel_size=hparams.conv_kernel,
                                           stride=1, padding=int((hparams.conv_kernel - 1) / 2),
                                           dilation=1, w_init_gain='linear')
        # pitch embedding
        self.pitch_embedding = ConvNorm1D(1, hparams.hidden_embed_dim, kernel_size=hparams.conv_kernel,
                                          stride=1, padding=int((hparams.conv_kernel - 1) / 2),
                                          dilation=1, w_init_gain='linear')
        # mel-spec pre-net convolutions
        self.convs = nn.Sequential(
            ConvNorm1D(nb_mels, hparams.conv_channels,
                       kernel_size=hparams.conv_kernel, stride=1,
                       padding=int((hparams.conv_kernel - 1) / 2),
                       dilation=1, w_init_gain='relu'),
            nn.ReLU(),
            nn.LayerNorm(hparams.conv_channels),
            nn.Dropout(hparams.conv_dropout),
            ConvNorm1D(hparams.conv_channels, hparams.conv_channels,
                       kernel_size=hparams.conv_kernel, stride=1,
                       padding=int((hparams.conv_kernel - 1) / 2),
                       dilation=1, w_init_gain='relu'),
            nn.ReLU(),
            nn.LayerNorm(hparams.conv_channels),
            nn.Dropout(hparams.conv_dropout),
            ConvNorm1D(hparams.conv_channels, hparams.hidden_embed_dim,
                       kernel_size=hparams.conv_kernel, stride=1,
                       padding=int((hparams.conv_kernel - 1) / 2),
                       dilation=1, w_init_gain='relu'),
            nn.ReLU(),
            nn.LayerNorm(hparams.hidden_embed_dim),
            nn.Dropout(hparams.conv_dropout)
        )
        # FFT blocks
        blocks = []
        for _ in range(hparams.nb_blocks):
            blocks.append(FFTBlock(hparams))
        self.blocks = nn.ModuleList(blocks)
        # speaker embedding
        self.spk_embedding = nn.Embedding(n_speakers, hparams.hidden_embed_dim)
        torch.nn.init.xavier_uniform_(self.spk_embedding.weight.data)
        # projection layers for FiLM parameters
        nb_tot_film_params = 0
        for _, module_params in self.module_params.items():
            nb_blocks, conv_channels = module_params
            nb_tot_film_params += nb_blocks * conv_channels
        self.gammas_predictor = LinearNorm(hparams.hidden_embed_dim, nb_tot_film_params, w_init_gain='linear')
        self.betas_predictor = LinearNorm(hparams.hidden_embed_dim, nb_tot_film_params, w_init_gain='linear')
        # initialize L2 penalized scalar post-multipliers
        # one (gamma, beta) scalar post-multiplier per FiLM layer, i.e per block
        if self.post_mult_weight != 0.:
            nb_post_multipliers = 0
            for _, module_params in self.module_params.items():
                nb_blocks, _ = module_params
                nb_post_multipliers += nb_blocks
            self.post_multipliers = Parameter(torch.empty(2, nb_post_multipliers))  # (2, nb_post_multipliers)
            nn.init.xavier_uniform_(self.post_multipliers, gain=nn.init.calculate_gain('linear'))  # (2, nb_post_multipliers)
        else:
            self.post_multipliers = 1.
    
    def forward(self, frames_energy, frames_pitch, mel_specs, speaker_ids, output_lengths):
        ''' Forward function of Prosody Encoder:
            frames_energy = (B, T_max)
            frames_pitch = (B, T_max)
            mel_specs = (B, nb_mels, T_max)
            speaker_ids = (B, )
            output_lengths = (B, )
        '''
        # compute positional encoding
        pos = self.pos_enc(output_lengths.unsqueeze(1))  # (B, T_max, hidden_embed_dim)
        # encode energy sequence
        frames_energy = frames_energy.unsqueeze(2)  # (B, T_max, 1)
        energy = self.energy_embedding(frames_energy)  # (B, T_max, hidden_embed_dim)
        # encode pitch sequence
        frames_pitch = frames_pitch.unsqueeze(2)  # (B, T_max, 1)
        pitch = self.pitch_embedding(frames_pitch)  # (B, T_max, hidden_embed_dim)
        # pass through convs
        mel_specs = mel_specs.transpose(1, 2)  # (B, T_max, nb_mels)
        outputs = self.convs(mel_specs)  # (B, T_max, hidden_embed_dim)
        # create mask
        mask = ~get_mask_from_lengths(output_lengths) # (B, T_max)
        # add encodings and mask tensor
        outputs = outputs + energy + pitch + pos  # (B, T_max, hidden_embed_dim)
        outputs = outputs.masked_fill(mask.unsqueeze(2), 0)  # (B, T_max, hidden_embed_dim)
        # pass through FFT blocks
        for _, block in enumerate(self.blocks):
            outputs = block(outputs, None, mask)  # (B, T_max, hidden_embed_dim)
        # average pooling on the whole time sequence
        outputs = torch.sum(outputs, dim=1) / output_lengths.unsqueeze(1)  # (B, hidden_embed_dim)
        # store prosody embeddings
        prosody_embeddings = outputs  # (B, hidden_embed_dim)
        # encode speaker IDs and add
        speaker_ids = self.spk_embedding(speaker_ids)  # (B, hidden_embed_dim)
        outputs = outputs + speaker_ids  # (B, hidden_embed_dim)
        
        # project outputs to predict all FiLM parameters
        gammas = self.gammas_predictor(outputs)  # (B, nb_tot_film_params)
        betas = self.betas_predictor(outputs)  # (B, nb_tot_film_params)
        # split FiLM parameters per FiLM-ed module
        modules_film_params = []
        column_idx, block_idx = 0, 0
        for _, module_params in self.module_params.items():
            nb_blocks, conv_channels = module_params
            module_nb_film_params = nb_blocks * conv_channels
            module_gammas = gammas[:, column_idx: column_idx + module_nb_film_params]  # (B, module_nb_film_params)
            module_betas = betas[:, column_idx: column_idx + module_nb_film_params]  # (B, module_nb_film_params)
            # split FiLM parameters for each block in the module
            B = module_gammas.size(0)
            module_gammas = module_gammas.view(B, nb_blocks, -1)  # (B, nb_blocks, block_nb_film_params)
            module_betas = module_betas.view(B, nb_blocks, -1)  # (B, nb_blocks, block_nb_film_params)
            # predict gammas in the delta regime, i.e. predict deviation from unity
            # add gamma scalar L2 penalized post-multiplier for each block
            if self.post_mult_weight != 0.:
                gamma_post = self.post_multipliers[0, block_idx: block_idx + nb_blocks]  # (nb_blocks, )
                gamma_post = gamma_post.unsqueeze(0).unsqueeze(-1)  # (1, nb_blocks, 1)
            else:
                gamma_post = self.post_multipliers
            module_gammas = gamma_post * module_gammas + 1  # (B, nb_blocks, block_nb_film_params)
            # add betas scalar L2 penalized post-multiplier for each block
            if self.post_mult_weight != 0.:
                beta_post = self.post_multipliers[1, block_idx: block_idx + nb_blocks]  # (nb_blocks, )
                beta_post = beta_post.unsqueeze(0).unsqueeze(-1)  # (1, nb_blocks, 1)
            else:
                beta_post = self.post_multipliers
            module_betas = beta_post * module_betas  # (B, nb_blocks, block_nb_film_params)
            # concatenate tensors and append to list
            module_film_params = torch.cat((module_gammas, module_betas), dim=2)  # (B, nb_blocks, nb_film_params)
            modules_film_params.append(module_film_params)
            # increment variables
            block_idx += nb_blocks
            column_idx += module_nb_film_params
        encoder_film, prosody_pred_film, decoder_film = modules_film_params
        
        return prosody_embeddings, encoder_film, prosody_pred_film, decoder_film


class PhonemeEncoder(nn.Module):
    ''' Phoneme Encoder Module:
        - Symbols Embedding
        - Positional Encoding
        - 4x FFT Blocks with FiLM conditioning
    '''
    def __init__(self, hparams):
        super(PhonemeEncoder, self).__init__()
        n_symbols = hparams.n_symbols
        embed_dim = hparams.phoneme_encoder['hidden_embed_dim']
        Tuple = namedtuple('Tuple', hparams.phoneme_encoder)
        hparams = Tuple(**hparams.phoneme_encoder)
        
        # symbols embedding and positional encoding
        self.symbols_embedding = nn.Embedding(n_symbols, embed_dim)
        torch.nn.init.xavier_uniform_(self.symbols_embedding.weight.data)
        self.pos_enc = PositionalEncoding(embed_dim)
        # FFT blocks
        blocks = []
        for _ in range(hparams.nb_blocks):
            blocks.append(FFTBlock(hparams))
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x, film_params, input_lengths):
        ''' Forward function of Phoneme Encoder:
            x = (B, L_max)
            film_params = (B, nb_blocks, nb_film_params)
            input_lengths = (B, )
        '''
        # compute symbols embedding
        x = self.symbols_embedding(x)  # (B, L_max, hidden_embed_dim)
        # compute positional encoding
        pos = self.pos_enc(input_lengths.unsqueeze(1))  # (B, L_max, hidden_embed_dim)
        # create mask
        mask = ~get_mask_from_lengths(input_lengths) # (B, L_max)
        # add and mask
        x = x + pos  # (B, L_max, hidden_embed_dim)
        x = x.masked_fill(mask.unsqueeze(2), 0)  # (B, L_max, hidden_embed_dim)
        # pass through FFT blocks
        for idx, block in enumerate(self.blocks):
            x = block(x, film_params[:, idx, :], mask)  # (B, L_max, hidden_embed_dim)
        
        return x


class LocalProsodyPredictor(nn.Module):
    ''' Local Prosody Predictor Module:
        - 2x Conv 1D
        - FiLM conditioning
        - Linear projection
    '''
    def __init__(self, hparams):
        super(LocalProsodyPredictor, self).__init__()
        embed_dim = hparams.phoneme_encoder['hidden_embed_dim']
        Tuple = namedtuple('Tuple', hparams.local_prosody_predictor)
        hparams = Tuple(**hparams.local_prosody_predictor)
        
        # conv1D blocks
        blocks = []
        for idx in range(hparams.nb_blocks):
            in_channels = embed_dim if idx == 0 else hparams.conv_channels
            convs =  nn.Sequential(
                ConvNorm1D(in_channels, hparams.conv_channels,
                           kernel_size=hparams.conv_kernel, stride=1,
                           padding=int((hparams.conv_kernel - 1) / 2),
                           dilation=1, w_init_gain='relu'),
                nn.ReLU(),
                nn.LayerNorm(hparams.conv_channels),
                nn.Dropout(hparams.conv_dropout),
                ConvNorm1D(hparams.conv_channels, hparams.conv_channels,
                           kernel_size=hparams.conv_kernel, stride=1,
                           padding=int((hparams.conv_kernel - 1) / 2),
                           dilation=1, w_init_gain='relu'),
                nn.ReLU(),
                nn.LayerNorm(hparams.conv_channels),
                nn.Dropout(hparams.conv_dropout)
            )
            blocks.append(convs)
        self.blocks = nn.ModuleList(blocks)
        # linear projection for prosody prediction
        self.projection = LinearNorm(hparams.conv_channels, 3, w_init_gain='linear')
        
    def forward(self, x, film_params, input_lengths):
        ''' Forward function of Local Prosody Predictor:
            x = (B, L_max, hidden_embed_dim)
            film_params = (B, nb_blocks, nb_film_params)
            input_lengths = (B, )
        '''
        # pass through blocks and mask tensor
        for idx, block in enumerate(self.blocks):
            x = block(x)  # (B, L_max, conv_channels)
            # add FiLM transformation
            block_film_params = film_params[:, idx, :]  # (B, nb_film_params)
            nb_gammas = int(block_film_params.size(1) / 2)
            assert(nb_gammas == x.size(2))
            gammas = block_film_params[:, :nb_gammas].unsqueeze(1)  # (B, 1, conv_channels)
            betas = block_film_params[:, nb_gammas:].unsqueeze(1)  # (B, 1, conv_channels)
            x = gammas * x + betas  # (B, L_max, conv_channels)
        mask = ~get_mask_from_lengths(input_lengths) # (B, L_max)
        x = x.masked_fill(mask.unsqueeze(2), 0)  # (B, L_max, conv_channels)
        # predict prosody params and mask tensor
        prosody_preds = self.projection(x)  # (B, L_max, 3)
        prosody_preds = prosody_preds.masked_fill(mask.unsqueeze(2), 0)  # (B, L_max, 3)
        # extract prosody params
        durations = prosody_preds[:, :, 0]  # (B, L_max)
        energies = prosody_preds[:, :, 1]  # (B, L_max)
        pitch = prosody_preds[:, :, 2]  # (B, L_max)
        
        return durations, energies, pitch


class GaussianUpsamplingModule(nn.Module):
    ''' Gaussian Upsampling Module:
        - Duration Projection
        - Energy Projection
        - Pitch Projection
        - Ranges Projection Layer
        - Gaussian Upsampling
    '''
    def __init__(self, hparams):
        super(GaussianUpsamplingModule, self).__init__()
        embed_dim = hparams.phoneme_encoder['hidden_embed_dim']
        Tuple = namedtuple('Tuple', hparams.gaussian_upsampling_module)
        hparams = Tuple(**hparams.gaussian_upsampling_module)
        
        # duration, energy and pitch projection layers
        self.duration_projection = ConvNorm1D(1, embed_dim, kernel_size=hparams.conv_kernel,
                                              stride=1, padding=int((hparams.conv_kernel - 1) / 2),
                                              dilation=1, w_init_gain='linear')
        self.energy_projection = ConvNorm1D(1, embed_dim, kernel_size=hparams.conv_kernel,
                                            stride=1, padding=int((hparams.conv_kernel - 1) / 2),
                                            dilation=1, w_init_gain='linear')
        self.pitch_projection = ConvNorm1D(1, embed_dim, kernel_size=hparams.conv_kernel,
                                           stride=1, padding=int((hparams.conv_kernel - 1) / 2),
                                           dilation=1, w_init_gain='linear')
        # ranges predictor
        self.projection = nn.Sequential(
            LinearNorm(embed_dim, 1, w_init_gain='relu'),
            nn.Softplus()
        )
    
    def forward(self, x, durations_float, durations_int, energies, pitch, input_lengths):
        ''' Forward function of Gaussian Upsampling Module:
            x = (B, L_max, hidden_embed_dim)
            durations_float = (B, L_max)
            durations_int = (B, L_max)
            energies = (B, L_max)
            pitch = (B, L_max)
            input_lengths = (B, )
        '''
        # project durations
        durations = durations_float.unsqueeze(2)  # (B, L_max, 1)
        durations = self.duration_projection(durations)  # (B, L_max, hidden_embed_dim)
        # project energies
        energies = energies.unsqueeze(2)  # (B, L_max, 1)
        energies = self.energy_projection(energies)  # (B, L_max, hidden_embed_dim)
        # project pitch
        pitch = pitch.unsqueeze(2)  # (B, L_max, 1)
        pitch = self.pitch_projection(pitch)  # (B, L_max, hidden_embed_dim)
        
        # add energy and pitch to encoded input symbols
        x = x + energies + pitch  # (B, L_max, hidden_embed_dim)
        
        # predict ranges for each symbol and mask tensor
        # use mask_value = 1. because ranges will be used as stds in Gaussian upsampling
        # mask_value = 0. would cause NaN values
        range_inputs = x + durations  # (B, L_max, hidden_embed_dim) 
        ranges = self.projection(range_inputs)  # (B, L_max, 1)
        ranges = ranges.squeeze(2)  # (B, L_max)
        mask = ~get_mask_from_lengths(input_lengths) # (B, L_max)
        ranges = ranges.masked_fill(mask, 1)  # (B, L_max)
        
        # perform Gaussian upsampling
        # compute Gaussian means
        means = durations_int.float() / 2  # (B, L_max)
        cumsum = torch.cumsum(durations_int, dim=1)  # (B, L_max)
        means[:, 1:] += cumsum[:, :-1]  # (B, L_max)
        # compute Gaussian distributions
        means = means.unsqueeze(-1)  # (B, L_max, 1)
        stds = ranges.unsqueeze(-1)  # (B, L_max, 1)
        gaussians = Normal(means, stds)  # (B, L_max, 1)
        # create frames idx tensor
        nb_frames_max = torch.max(cumsum)  # T_max
        frames_idx = torch.FloatTensor([i + 0.5 for i in range(nb_frames_max)])  # (T_max, )
        frames_idx = frames_idx.cuda(x.device, non_blocking=True).float()  # (T_max, )
        # compute probs
        probs = torch.exp(gaussians.log_prob(frames_idx))  # (B, L_max, T_max)
        # apply mask to set probs out of sequence length to 0
        probs = probs.masked_fill(mask.unsqueeze(2), 0)  # (B, L_max, T_max)
        # compute weights
        weights = probs / (torch.sum(probs, dim=1, keepdim=True) + 1e-20)  # (B, L_max, T_max)
        # compute upsampled embedding
        x_upsamp = torch.sum(x.unsqueeze(-1) * weights.unsqueeze(2), dim=1)  # (B, input_dim, T_max)
        x_upsamp = x_upsamp.permute(0, 2, 1)  # (B, T_max, input_dim)
        
        return x_upsamp, weights


class FrameDecoder(nn.Module):
    ''' Frame Decoder Module:
        - Positional Encoding
        - 4x FFT Blocks with FiLM conditioning
        - Linear projection
    '''
    def __init__(self, hparams):
        super(FrameDecoder, self).__init__()
        nb_mels = hparams.n_mel_channels
        embed_dim = hparams.phoneme_encoder['hidden_embed_dim']
        hparams.frame_decoder['hidden_embed_dim'] = embed_dim
        Tuple = namedtuple('Tuple', hparams.frame_decoder)
        hparams = Tuple(**hparams.frame_decoder)
        
        # positional encoding
        self.pos_enc = PositionalEncoding(embed_dim)
        # FFT blocks
        blocks = []
        for _ in range(hparams.nb_blocks):
            blocks.append(FFTBlock(hparams))
        self.blocks = nn.ModuleList(blocks)
        # linear projection for mel-spec prediction
        self.projection = LinearNorm(embed_dim, nb_mels, w_init_gain='linear')
    
    def forward(self, x, film_params, output_lengths):
        ''' Forward function of Decoder Embedding:
            x = (B, T_max, hidden_embed_dim)
            film_params = (B, nb_blocks, nb_film_params)
            output_lengths = (B, )
        '''
        # compute positional encoding
        pos = self.pos_enc(output_lengths.unsqueeze(1))  # (B, T_max, hidden_embed_dim)
        # create mask
        mask = ~get_mask_from_lengths(output_lengths) # (B, T_max)
        # add and mask
        x = x + pos  # (B, T_max, hidden_embed_dim)
        x = x.masked_fill(mask.unsqueeze(2), 0)  # (B, T_max, hidden_embed_dim)
        # pass through FFT blocks
        for idx, block in enumerate(self.blocks):
            x = block(x, film_params[:, idx, :], mask)  # (B, T_max, hidden_embed_dim)
        # predict mel-spec frames and mask tensor
        mel_specs = self.projection(x)  # (B, T_max, nb_mels)
        mel_specs = mel_specs.masked_fill(mask.unsqueeze(2), 0)  # (B, T_max, nb_mels)
        mel_specs = mel_specs.transpose(1, 2)  # (B, nb_mels, T_max)
        
        return mel_specs


class DaftExprt(nn.Module):
    ''' DaftExprt model from J. Zaïdi, H. Seuté, B. van Niekerk, M.A. Carbonneau
        "DaftExprt: Robust Prosody Transfer Across Speakers for Expressive Speech Synthesis"
        arXiv:2108.02271, 2021.
    '''
    def __init__(self, hparams):
        super(DaftExprt, self).__init__()
        self.prosody_encoder = ProsodyEncoder(hparams)
        self.speaker_classifier = SpeakerClassifier(hparams)
        self.phoneme_encoder = PhonemeEncoder(hparams)
        self.prosody_predictor = LocalProsodyPredictor(hparams)
        self.gaussian_upsampling = GaussianUpsamplingModule(hparams)
        self.frame_decoder = FrameDecoder(hparams)
    
    def parse_batch(self, gpu, batch):
        ''' Parse input batch
        '''
        # extract tensors
        symbols, durations_float, durations_int, symbols_energy, symbols_pitch, input_lengths, \
            frames_energy, frames_pitch, mel_specs, output_lengths, speaker_ids, feature_dirs, feature_files = batch
        
        # transfer tensors to specified GPU
        symbols = symbols.cuda(gpu, non_blocking=True).long()                        # (B, L_max)
        durations_float = durations_float.cuda(gpu, non_blocking=True).float()       # (B, L_max)
        durations_int = durations_int.cuda(gpu, non_blocking=True).long()            # (B, L_max)
        symbols_energy = symbols_energy.cuda(gpu, non_blocking=True).float()         # (B, L_max)
        symbols_pitch = symbols_pitch.cuda(gpu, non_blocking=True).float()           # (B, L_max)
        input_lengths = input_lengths.cuda(gpu, non_blocking=True).long()            # (B, )
        frames_energy = frames_energy.cuda(gpu, non_blocking=True).float()           # (B, T_max)
        frames_pitch = frames_pitch.cuda(gpu, non_blocking=True).float()             # (B, T_max)
        mel_specs = mel_specs.cuda(gpu, non_blocking=True).float()                   # (B, n_mel_channels, T_max)
        output_lengths = output_lengths.cuda(gpu, non_blocking=True).long()          # (B, )
        speaker_ids = speaker_ids.cuda(gpu, non_blocking=True).long()                # (B, )
        
        # create inputs and targets
        inputs = (symbols, durations_float, durations_int, symbols_energy, symbols_pitch, input_lengths,
                  frames_energy, frames_pitch, mel_specs, output_lengths, speaker_ids)
        targets = (durations_float, symbols_energy, symbols_pitch, mel_specs, speaker_ids)
        file_ids = (feature_dirs, feature_files)
        
        return inputs, targets, file_ids
    
    def forward(self, inputs):
        ''' Forward function of DaftExprt
        '''
        # extract inputs
        symbols, durations_float, durations_int, symbols_energy, symbols_pitch, input_lengths, \
            frames_energy, frames_pitch, mel_specs, output_lengths, speaker_ids = inputs
        input_lengths, output_lengths = input_lengths.detach(), output_lengths.detach()
        
        # extract FiLM parameters from reference and speaker ID
        # (B, nb_blocks, nb_film_params)
        prosody_embed, encoder_film, prosody_pred_film, decoder_film = self.prosody_encoder(frames_energy, frames_pitch, mel_specs, speaker_ids, output_lengths)
        # pass through speaker classifier
        spk_preds = self.speaker_classifier(prosody_embed)  # (B, nb_speakers)
        # embed phoneme symbols, add positional encoding and encode input sequence
        enc_outputs = self.phoneme_encoder(symbols, encoder_film, input_lengths)  # (B, L_max, hidden_embed_dim)
        # predict prosody parameters
        duration_preds, energy_preds, pitch_preds = self.prosody_predictor(enc_outputs, prosody_pred_film, input_lengths)  # (B, L_max)
        # perform Gaussian upsampling on symbols sequence
        # use prosody ground-truth values for training
        # symbols_upsamp = (B, T_max, hidden_embed_dim)
        # weights = (B, L_max, T_max)
        symbols_upsamp, weights = self.gaussian_upsampling(enc_outputs, durations_float, durations_int, symbols_energy, symbols_pitch, input_lengths)
        # decode output sequence and predict mel-specs
        mel_spec_preds = self.frame_decoder(symbols_upsamp, decoder_film, output_lengths)  # (B, nb_mels, T_max)
        
        # parse outputs
        speaker_preds = spk_preds
        film_params = [self.prosody_encoder.post_multipliers, encoder_film, prosody_pred_film, decoder_film]
        encoder_preds = [duration_preds, energy_preds, pitch_preds, input_lengths]
        decoder_preds = [mel_spec_preds, output_lengths]
        alignments = weights
        
        return speaker_preds, film_params, encoder_preds, decoder_preds, alignments
    
    def get_int_durations(self, duration_preds, hparams):
        ''' Convert float durations to integer frame durations
        '''
        # min float duration to have at least one mel-spec frame attributed to the symbol
        fft_length = hparams.filter_length / hparams.sampling_rate
        dur_min = fft_length / 2
        # set duration under min duration to 0.
        duration_preds[duration_preds < dur_min] = 0.  # (B, L_max)
        # convert to int durations for each element in the batch
        durations_int = torch.LongTensor(duration_preds.size(0), duration_preds.size(1)).zero_()  # (B, L_max)
        for line_idx in range(duration_preds.size(0)):
            end_prev, symbols_idx, durations_float = 0., [], []
            for symbol_id in range(duration_preds.size(1)):
                symb_dur = duration_preds[line_idx, symbol_id].item()
                if symb_dur != 0.:  # ignore 0 durations
                    symbols_idx.append(symbol_id)
                    durations_float.append([end_prev, end_prev + symb_dur])
                    end_prev += symb_dur
            int_durs = torch.LongTensor(duration_to_integer(durations_float, hparams))  # (L_max, )
            durations_int[line_idx, symbols_idx] = int_durs
        # put on GPU
        durations_int = durations_int.cuda(duration_preds.device, non_blocking=True).long()  # (B, L_max)
        
        return duration_preds, durations_int
    
    def pitch_shift(self, pitch_preds, pitch_factors, hparams, speaker_ids):
        ''' Pitch shift pitch predictions
            Pitch factors are assumed to be in Hz
        '''
        # keep track of unvoiced idx
        zero_idxs = (pitch_preds == 0.).nonzero()  # (N, 2)
        # pitch factors are F0 shifts in Hz
        # pitch_factors = [[+50, -20, ...], ..., [+30, -10, ...]]
        for line_idx in range(pitch_preds.size(0)):
            speaker_id = speaker_ids[line_idx].item()
            pitch_mean = hparams.stats[f'spk {speaker_id}']['pitch']['mean']
            pitch_std = hparams.stats[f'spk {speaker_id}']['pitch']['std']
            pitch_preds[line_idx] = torch.exp(pitch_std * pitch_preds[line_idx] + pitch_mean)  # (L_max)
            # perform pitch shift in Hz domain
            pitch_preds[line_idx] += pitch_factors[line_idx]  # (L_max)
            # go back to log and re-normalize using pitch training stats
            pitch_preds[line_idx] = (torch.log(pitch_preds[line_idx]) - pitch_mean) / pitch_std  # (L_max)
        # set unvoiced idx to zero
        pitch_preds[zero_idxs[:, 0], zero_idxs[:, 1]] = 0.
        
        return pitch_preds
    
    def pitch_multiply(self, pitch_preds, pitch_factors):
        ''' Apply multiply transform to pitch prediction with respect to the mean

            Effects of factor values on the pitch:
                ]0, +inf[       amplify
                0               no effect
                ]-1, 0[         de-amplify
                -1              flatten
                ]-2, -1[        invert de-amplify
                -2              invert
                ]-inf, -2[      invert amplify
        '''
        # multiply pitch for each element in the batch
        for line_idx in range(pitch_preds.size(0)):
            # keep track of voiced and unvoiced idx
            non_zero_idxs = pitch_preds[line_idx].nonzero()  # (M, )
            zero_idxs = (pitch_preds[line_idx] == 0.).nonzero()  # (N, )
            # compute mean of voiced values
            mean_pitch = torch.mean(pitch_preds[line_idx, non_zero_idxs])
            # compute deviation to the mean for each pitch prediction
            pitch_deviation = pitch_preds[line_idx] - mean_pitch  # (L_max)
            # multiply factors to pitch deviation
            pitch_deviation *= pitch_factors[line_idx]  # (L_max)
            # add deviation to pitch predictions
            pitch_preds[line_idx] += pitch_deviation  # (L_max)
            # reset unvoiced values to 0
            pitch_preds[line_idx, zero_idxs] = 0.
        
        return pitch_preds
    
    def inference(self, inputs, pitch_transform, hparams):
        ''' Inference function of DaftExprt
        '''
        # symbols = (B, L_max)
        # dur_factors = (B, L_max)
        # energy_factors = (B, L_max)
        # pitch_factors = (B, L_max)
        # input_lengths = (B, )
        # energy_refs = (B, T_max)
        # pitch_refs = (B, T_max)
        # mel_spec_refs = (B, n_mel_channels, T_max)
        # ref_lengths = (B, )
        # speaker_ids = (B, )
        symbols, dur_factors, energy_factors, pitch_factors, input_lengths, \
            energy_refs, pitch_refs, mel_spec_refs, ref_lengths, speaker_ids = inputs
        
        # extract FiLM parameters from reference and speaker ID
        # (B, nb_blocks, nb_film_params)
        _, encoder_film, prosody_pred_film, decoder_film = self.prosody_encoder(energy_refs, pitch_refs, mel_spec_refs, speaker_ids, ref_lengths)
        # embed phoneme symbols, add positional encoding and encode input sequence
        enc_outputs = self.phoneme_encoder(symbols, encoder_film, input_lengths)  # (B, L_max, hidden_embed_dim)
        # predict prosody parameters
        duration_preds, energy_preds, pitch_preds = self.prosody_predictor(enc_outputs, prosody_pred_film, input_lengths)  # (B, L_max)
        
        # multiply durations by duration factors and extract int durations
        duration_preds *= dur_factors  # (B, L_max)
        duration_preds, durations_int = self.get_int_durations(duration_preds, hparams)  # (B, L_max)
        # add energy factors to energies
        # set 0 energy for symbols with 0 duration
        energy_preds *= energy_factors  # (B, L_max)
        energy_preds[durations_int == 0] = 0.  # (B, L_max)
        # set unvoiced pitch for symbols with 0 duration
        # apply pitch factors using specified transformation
        pitch_preds[durations_int == 0] = 0.
        if pitch_transform == 'add':
            pitch_preds = self.pitch_shift(pitch_preds, pitch_factors, hparams, speaker_ids)  # (B, L_max)
        elif pitch_transform == 'multiply':
            pitch_preds = self.pitch_multiply(pitch_preds, pitch_factors)  # (B, L_max)
        else:
            raise NotImplementedError
        
        # perform Gaussian upsampling on symbols sequence
        # symbols_upsamp = (B, T_max, hidden_embed_dim)
        # weights = (B, L_max, T_max)
        symbols_upsamp, weights = self.gaussian_upsampling(enc_outputs, duration_preds, durations_int, energy_preds, pitch_preds, input_lengths)
        # get sequence output length for each element in the batch
        output_lengths = torch.sum(durations_int, dim=1)  # (B, )
        output_lengths = output_lengths.cuda(symbols_upsamp.device, non_blocking=True).long()  # (B, )
        assert(torch.max(output_lengths) == symbols_upsamp.size(1))
        # decode output sequence and predict mel-specs
        mel_spec_preds = self.frame_decoder(symbols_upsamp, decoder_film, output_lengths)  # (B, nb_mels, T_max)
        
        # parse outputs
        encoder_preds = [duration_preds, durations_int, energy_preds, pitch_preds, input_lengths]
        decoder_preds = [mel_spec_preds, output_lengths]
        alignments = weights
        
        return encoder_preds, decoder_preds, alignments
