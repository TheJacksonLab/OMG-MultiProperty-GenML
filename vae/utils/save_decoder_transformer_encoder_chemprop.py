import os
import pickle


class VAEParameters(object):
    def __init__(self, save_directory, eos_idx, latent_dimension,
                 encoder_feature_dim,
                 encoder_layer_1d, encoder_layer_2d, encoder_layer_3d, mpn_bias, mpn_depth, mpn_dropout_rate,
                 decoder_num_layers, decoder_num_attention_heads, max_length: int, vocab: dict,
                 decoder_loss_weights, decoder_add_latent,
                 property_dim,
                 property_network_hidden_dim_list: list, property_weights: tuple, property_linear: bool, dtype,
                 random_state=42):

        self.save_directory = save_directory
        self.eos_idx = eos_idx
        self.latent_dimension = latent_dimension
        self.encoder_feature_dim = encoder_feature_dim
        self.encoder_layer_1d = encoder_layer_1d
        self.encoder_layer_2d = encoder_layer_2d
        self.encoder_layer_3d = encoder_layer_3d
        self.mpn_bias = mpn_bias
        self.mpn_depth = mpn_depth
        self.mpn_dropout_rate = mpn_dropout_rate
        self.decoder_num_layers = decoder_num_layers
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.max_length = max_length,
        self.vocab = vocab,
        self.decoder_loss_weights = decoder_loss_weights,
        self.decoder_add_latent = decoder_add_latent,
        self.property_dim = property_dim
        self.property_network_hidden_dim_list = property_network_hidden_dim_list
        self.property_weights = property_weights
        self.dtype = dtype
        self.random_state = random_state
        self.property_linear = property_linear
