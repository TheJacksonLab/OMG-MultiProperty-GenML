import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('/Users/hwankim/PycharmProjects/OMG-MultiProperty-GenML/vae/decoder_transformer')
from model.embeddings_mod import Embeddings
from model.transformer_mod import TransformerDecoder

from onmt.translate import BeamSearch, GNMTGlobalScorer, GreedySearch
from onmt.modules import AverageAttention


## Transformer decoder
class SequenceDecoder(nn.Module):
    def __init__(self, embedding_dim, decoder_num_layers, num_attention_heads, max_length, vocab, loss_weights=None, add_latent=True,
                 cross_entropy_loss_reduction='mean'):
        """Implementation of transformer decoder. This is modified from https://github.com/Intelligent-molecular-systems/Inverse_copolymer_design/tree/main.

        Args:
            embedding_dim (int): word_vec_size (size of embedding vectors). This is the same as the latent dimension (equals word embedding dimension in this model).
            decoder_num_layers (int): number of Transformer decoder layers.
            num_attention_heads (int): number of multi-headed attention. The "d_model" is divided by the num_attention_heads where "d_model" is equal to the sum of word embedding vectors and the latent vector.
            max_length (int): Longest acceptable sequence, not counting begin-of-sentence (presumably there has been no EOS yet if max_length is used as a cutoff).
            vocab (dict): complete vocab of tokens. Ex) vocab (dict) -> ex) {‘_PAD’: 0, ’N’: 25, ..}
            loss_weights (tensor): weights to use in cross entropy loss. default: None
            add_latent (bool): whether to add the latent space embeddings to word embeddings. default: True

        The implementation of "PositionwiseFeedForward"
            self.w_1 = nn.Linear(d_model, d_ff)
            self.w_2 = nn.Linear(d_ff, d_model)
        """
        super().__init__()
        self.max_n = max_length
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.add_latent = add_latent

        # Embeddings(nn.Module) -> neural network.
        self.decoder_embeddings = Embeddings(
            word_vec_size=embedding_dim,  # size of embedding vectors
            word_vocab_size=len(self.vocab),  # size of dictionary of embeddings for words.
            word_padding_idx=self.vocab["[PAD]"],  # padding index for words in the embeddings. The embedding vector at the padding idx is not updated.
            position_encoding=False,  # whether to add positional encoding to word embedding.
            dropout=0.0  # dropout probability.
        )

        # add latent vectors to the word embeddings. d_model -> initial dimension used in masked multi-headed attention for Key, Value, and Query.
        if self.add_latent:
            d_model = embedding_dim * 2
        else:
            d_model = embedding_dim

        # TransformerDecoder (with EDATT) or TransformerLMDecoder (without EDAtt)
        self.Decoder = TransformerDecoder(
            num_layers=decoder_num_layers,  # number of decoder layers
            d_model=d_model,  # size of the model
            heads=num_attention_heads,  # number of multi-headed attention
            d_ff=2048,  # d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`. PositionwiseFeedForward is a two-layer Feed-Forward-Network with residual layer norm.
            copy_attn=False,  # if using a separate copy attention.
            self_attn_type="scaled-dot",  # (str): type of self-attention scaled-dot, average
            dropout=0.0,  # dropout (float): dropout in residual, self-attn(dot) and feed-forward
            attention_dropout=0.0,  # attention_dropout (float): dropout in context_attn (and self-attn(avg))
            embeddings=self.decoder_embeddings,  # embeddings (onmt.modules.Embeddings): embeddings to use, should have positional encodings
            max_relative_positions=0,  # (int): Max distance between inputs in relative positions representations. default: 0. "4" was used in the paper.
            relative_positions_buckets=0,  # default: 0
            aan_useffn=False,  # Turn on the FFN layer in the AAN decoder. -> default: False
            full_context_alignment=False,  # whether enable an extra full context decoder forward for alignment -> default: False
            alignment_layer=None,  #  N° Layer to supervise with for alignment guiding.  # originally -3? was used
            alignment_heads=0  # N. of cross attention heads to use for alignment guiding -> default: 0
        )

        # project to output logit
        self.output_layer = nn.Linear(d_model, len(self.vocab), bias=True)

        # cross entropy loss. Excluded "FocalLoss" in the original implementation.
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.vocab["[PAD]"],  # Specifies a target value that is ignored and does not contribute to the input gradient.
            reduction=cross_entropy_loss_reduction,
            weight=loss_weights
        )

    def forward(self, z, integer_encoded):
        """Forward pass of decoder

        Args:
            z (Tensor): Latent space embedding [b, h]
            integer_encoded (Tensor): Tensor containing integer encoded sentences [b, L]

            # b -> batch size
            # h -> latent space dimension
            # L -> max length

        Returns:
            Tensor, Tensor: Reconstruction loss and accuracy
        """
        z_length = z.size(1)
        src_lengths = torch.ones(z.size(0), device=z.device).long() * z_length  # long tensor [b,]

        # prepare target
        target = integer_encoded[:, :-1]  # pop last token -> shifted right.
        m = nn.ConstantPad1d((1, 0), self.vocab["[SOS]"])  # pads SOS token left side (beginning of sequences)
        target = m(target)
        target = target.unsqueeze(-1)

        # output from encoder. "src" means the output from encoder.
        enc_output = z.unsqueeze(1)
        self.Decoder.state["src"] = enc_output

        # decode. dec_attn -> dict. Has a key "std". -> dec_attn['std'] shape -> [b, t, 1]. dec_attn -> all 1 (?)
        dec_outs, dec_attn = self.Decoder(
            tgt=target,
            enc_out=enc_output,
            src_len=src_lengths,
            step=target.size(1),
            add_latent=self.add_latent
        )
        dec_outs = self.output_layer(dec_outs)  # [b, t, h] => [b, t, v]
        dec_outs = dec_outs.permute(0, 2, 1)  # [b, t, v] => [b, v, t]. v -> voca

        # evaluate
        # reproducible on GPU requires reduction none in CE loss in order to use torch.use_deterministic_algorithms(True), mean afterwards
        # https://discuss.pytorch.org/t/pytorchs-non-deterministic-cross-entropy-loss-and-the-problem-of-reproducibility/172180/8
        target = integer_encoded  # [b, t]

        recon_loss = self.criterion(
            input=dec_outs,  # (b, v, L)
            target=target  # (b, L)
        )

        # custom: assign more weight to predictions of critical decisions in string (stoichiometry and connectivity)
        # positional_weights = self.conditional_position_weights(torch.tensor(graph_batch.tgt_token_ids, device=z.device))
        # positional_weights = torch.from_numpy(positional_weights).to(z.device)

        # recon_loss = recon_loss * positional_weights
        # recon_loss = recon_loss.mean()

        predictions = torch.argmax(dec_outs.transpose(1, 0), dim=0)  # [b, t]. t -> sentence length.
        mask = (target != self.vocab["[PAD]"]).long()
        accs = (predictions == target).float()
        accs = accs * mask
        acc = accs.sum() / mask.sum()

        return recon_loss, acc, predictions, target

    # from onmt/translate/translator.py
    def inference(self, z, beam_size=1):
        global_scorer = GNMTGlobalScorer(alpha=0.5, beta=0.0, length_penalty="none", coverage_penalty="none")

        """ 
        BeamSearch
        beam_size (int): Number of beams to use (see base ``parallel_paths``).
        batch_size (int): See base.
        pad (int): See base.
        bos (int): See base.
        eos (int): See base.
        unk (int): See base.
        start (int): See base.
        n_best (int): Don't stop until at least this many beams have
            reached EOS.
        global_scorer (onmt.translate.GNMTGlobalScorer): Scorer instance.
        min_length (int): See base.
        max_length (int): See base.
        return_attention (bool): See base.
        block_ngram_repeat (int): See base.
        exclusion_tokens (set[int]): See base.
        ) 
        """
        decode_strategy = BeamSearch(
            pad=self.vocab["[PAD]"],
            bos=self.vocab["[SOS]"],
            eos=self.vocab["[EOS]"],
            unk=self.vocab["[UNK]"],
            ban_unk_token=False,  # Whether unk token is forbidden
            global_scorer=global_scorer,
            beam_size=beam_size,  # after the inference, it seems that the model shape is changed.. -> inference -> training -> error.
            start=self.vocab["[SOS]"],  # can be either bos or eos token
            batch_size=z.size(0),
            min_length=1,
            n_best=beam_size,
            stepwise_penalty=None,
            ratio=0.0,
            max_length=self.max_n,
            block_ngram_repeat=0,
            exclusion_tokens=set(),
            return_attention=False,
        )

        # decode_strategy = GreedySearch(
        #     pad=self.vocab["[PAD]"],
        #     bos=self.vocab["[SOS]"],
        #     eos=self.vocab["[EOS]"],
        #     unk=self.vocab["[UNK]"],
        #     ban_unk_token=True, # TODO: Check if true
        #     global_scorer=global_scorer,
        #     keep_topp=1,
        #     beam_size=beam_size,
        #     start=self.vocab["[SOS]"], # can be either bos or eos token
        #     batch_size=z.size(0),
        #     min_length=1,
        #     n_best=1,
        #     max_length=self.max_n,
        #     block_ngram_repeat=0,
        #     exclusion_tokens=set(),
        #     return_attention=False,
        #     sampling_temp=0.0,
        #     keep_topk=1
        # )

        # adapted from onmt.translate.translator
        # (0) Prep the components of the search.
        parallel_paths = decode_strategy.parallel_paths  # beam_size

        # (1) Encoder output.
        z_length = z.size(1)
        src_lengths = torch.ones(z.size(0), device=z.device).long() * z_length  # long tensor [b,]
        enc_output = z.unsqueeze(1)
        self.Decoder.state["src"] = enc_output  # init state

        # (2) prep decode_strategy. Possibly repeat src objects.
        src_map = None  # None
        target_prefix = None  # None

        fn_map_state, enc_out, src_map = decode_strategy.initialize(
            enc_out=enc_output,
            src_len=src_lengths,
            src_map=src_map,
            target_prefix=target_prefix,
            device=z.device
        )

        if fn_map_state is not None:
            self.Decoder.map_state(fn_map_state)

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            # decode
            decoder_input = decode_strategy.current_predictions.view(-1, 1, 1)

            # adapted from _decode_and_generate under Inference class (onmt/translate/translator)
            dec_outs, dec_attn = self.Decoder(  # dec_attn['std'] -> [b, 1, 1]
                tgt=decoder_input,
                enc_out=enc_out,
                src_len=decode_strategy.src_len,
                step=step,
                add_latent=self.add_latent
            )

            # get attention - for not self.copy_attn
            if "std" in dec_attn:
                attn = dec_attn["std"].detach()
            else:
                attn = None

            # calculate scores & advance
            scores = self.output_layer(dec_outs.squeeze(1))
            log_probs = F.log_softmax(scores.to(torch.float32), dim=-1).detach()
            decode_strategy.advance(log_probs, attn)

            # any_finished = decode_strategy.is_finished_list.any()  # there is no attribute of "is_finished".
            any_finished = any(
                [any(sublist) for sublist in decode_strategy.is_finished_list]
            )
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished:
                # Reorder states.
                if isinstance(enc_out, tuple):
                    enc_out = tuple(x[select_indices] for x in enc_out)
                else:
                    enc_out = enc_out[select_indices]

                if src_map is not None:
                    src_map = src_map[select_indices]

            if parallel_paths > 1 or any_finished:
                self.Decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices)
                )

        self.reset_layer_cache()  # reset layers 

        return decode_strategy.predictions

    def reset_layer_cache(self):
        """After inference, layer cache needs to be reset"""
        for layer in self.Decoder.transformer_layers:
            if isinstance(layer.self_attn, AverageAttention):
                layer.self_attn.layer_cache = False, {"prev_g": torch.tensor([])}
            else:
                layer.self_attn.layer_cache = (
                    False,
                    {"keys": torch.tensor([]), "values": torch.tensor([])},
                )
            layer.context_attn.layer_cache = (
                False,
                {"keys": torch.tensor([]), "values": torch.tensor([])},
            )

