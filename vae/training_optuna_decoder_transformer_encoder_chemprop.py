import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

import sys

from math import ceil
from tqdm import tqdm
from pathlib import Path

from torch.nn import functional as f
from torch.optim.lr_scheduler import StepLR, ExponentialLR, LambdaLR

from matplotlib.lines import Line2D
from rdkit.Chem import MolFromSmiles
from sklearn.metrics import r2_score
from vae.utils.save import save_model

CHEMPROP_PATH = '/home/sk77/PycharmProjects/omg_multi_task/vae/encoder_chemprop/chemprop'
sys.path.append(CHEMPROP_PATH)
from chemprop.data import MoleculeDataset


class EarlyStopping(object):
    def __init__(self, tolerance=5, min_delta=1e-5):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.previous_validation_loss = 1e31
        self.early_stop = False

    def __call__(self, validation_loss):
        if validation_loss > (self.previous_validation_loss + self.min_delta):
            self.counter += 1
        else:
            self.counter = 0

        if self.counter >= self.tolerance:
            self.early_stop = True

        # change
        self.previous_validation_loss = validation_loss


def train_model(vae_encoder, vae_decoder, property_predictor, data_train, data_valid, integer_encoded_train,
                integer_encoded_valid, y_train, y_valid, y_scaler, num_epochs, batch_size, lr_property, lr_enc,
                lr_dec, save_directory, check_point_dir, grammar_fragment, dtype, device, weight_decay=1e-5,
                mmd_weight=10.0, property_weight=1.0):
    """
    Train the Variational Auto-Encoder without DataDistributedParallel
    """
    print('num_epochs: ', num_epochs, flush=True)

    # set optimizer
    optimizer_encoder = torch.optim.Adam(vae_encoder.parameters(), lr=lr_enc, weight_decay=weight_decay)
    optimizer_decoder = torch.optim.Adam(vae_decoder.parameters(), lr=lr_dec, weight_decay=weight_decay)
    optimizer_property_predictor = torch.optim.Adam(property_predictor.parameters(), lr=lr_property, weight_decay=weight_decay)

    # plot lr rates
    lr_plot = list()

    # learning rate scheduler - exponential lr
    # step_size = 40
    # gamma = 0.1
    # encoder_lr_scheduler = ExponentialLR(optimizer=optimizer_encoder, gamma=gamma)
    # decoder_lr_scheduler = ExponentialLR(optimizer=optimizer_decoder, gamma=gamma)
    # property_predictor_lr_scheduler = ExponentialLR(optimizer=optimizer_property_predictor, gamma=gamma)

    # learning rate scheduler - gamma lr
    def warmup(current_epoch: int):
        """
        This function is needed for a lambda scheduler. current_epoch starts from zero. The peak value (lr) is used twice during training.
        """
        if current_epoch < lr_dict['warmup_epochs']:  # current_step / warmup_steps * base_lr
            return float((current_epoch + 1) / lr_dict['warmup_epochs'])
        else:  # (num_training_steps - current_step) / (num_training_steps - warmup_steps) * base_lr
            return max(0.0, float(lr_dict['training_epochs'] - current_epoch) / float(max(1, lr_dict['training_epochs'] - lr_dict['warmup_epochs'])))

    # warmup lr
    warmup_ratio = 0.1
    lr_dict = {'warmup_epochs': int(num_epochs * warmup_ratio), 'training_epochs': num_epochs}
    encoder_lr_scheduler = LambdaLR(optimizer=optimizer_encoder, lr_lambda=warmup)  # warm up schedular. The learning rate is reduced by a multiplicative factor.
    decoder_lr_scheduler = LambdaLR(optimizer=optimizer_decoder, lr_lambda=warmup)  # warm up schedular. The learning rate is reduced by a multiplicative factor.
    property_predictor_lr_scheduler = LambdaLR(optimizer=optimizer_property_predictor, lr_lambda=warmup)  # warm up schedular. The learning rate is reduced by a multiplicative factor.

    # train and valid data
    num_batches_train = ceil(len(data_train) / batch_size)
    num_batches_valid = ceil(len(data_valid) / batch_size)

    data_train_reconstruction_loss_list_teacher_forcing, data_train_divergence_loss_list_teacher_forcing, data_train_property_loss_list_teacher_forcing = [], [], []
    data_valid_reconstruction_loss_list_teacher_forcing, data_valid_divergence_loss_list_teacher_forcing, data_valid_property_loss_list_teacher_forcing = [], [], []

    # early stopping
    early_stopping = EarlyStopping(tolerance=5, min_delta=1e-10)

    # training
    for epoch in range(num_epochs):
        start = time.time()
        train_reconstruction_loss = torch.zeros(size=(1,), dtype=dtype).cpu()
        train_divergence_loss = torch.zeros(size=(1,), dtype=dtype).cpu()
        train_property_loss = torch.zeros(size=(1,), dtype=dtype).cpu()
        tqdm.write("==== Epoch %d Training" % (epoch + 1))

        # lr plot
        lr_plot.append(encoder_lr_scheduler.get_last_lr()[0])
        assert encoder_lr_scheduler.get_last_lr()[0] == decoder_lr_scheduler.get_last_lr()[0]
        assert decoder_lr_scheduler.get_last_lr()[0] == property_predictor_lr_scheduler.get_last_lr()[0]

        # change mode
        vae_encoder.train()
        vae_decoder.train()
        property_predictor.train()

        # random permutation
        rand_idx = torch.randperm(len(data_train))
        y_train_epoch = y_train.clone()[rand_idx]

        # mini-batch training
        tqdm.write("Training ...")
        pbar = tqdm(range(num_batches_train), total=num_batches_train, leave=True)
        with pbar as t:
            for batch_iteration in t:
                # manual batch iterations
                start_idx = batch_iteration * batch_size
                stop_idx = (batch_iteration + 1) * batch_size

                if batch_iteration == num_batches_train - 1:
                    stop_idx = len(data_train)

                # get batch data
                data_train_batch = MoleculeDataset([data_train[idx] for idx in rand_idx[start_idx: stop_idx]])
                integer_encoded_train_batch = integer_encoded_train[rand_idx[start_idx: stop_idx]].to(device)
                y_train_batch = y_train_epoch[start_idx: stop_idx].to(device)

                # Chemprop encoder
                latent_points, mus, log_vars = vae_encoder(data_train_batch)

                criterion = nn.MSELoss(reduction='none').to(device)
                predicted_train_property = property_predictor(latent_points)

                # compute train loss (elementwise multiplication)
                property_loss = criterion(input=predicted_train_property,
                                          target=y_train_batch)  # (num_batch, num_properties)
                property_loss = torch.sum(
                    torch.mean(property_loss, dim=0) * property_predictor.normalized_weights_tensor.to(device),
                    dim=0
                )
                weighted_property_loss = property_loss * property_weight

                # latent points (num_molecules_batch, latent_dim)
                reconstruction_loss, acc, predictions, target = vae_decoder(latent_points, integer_encoded_train_batch)

                # loss
                divergence_loss = estimate_maximum_mean_discrepancy(posterior_z_samples=latent_points)
                weighted_divergence_loss = mmd_weight * divergence_loss

                total_train_loss = reconstruction_loss + weighted_divergence_loss + weighted_property_loss

                # perform back propagation
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()
                optimizer_property_predictor.zero_grad()

                total_train_loss.backward()

                optimizer_encoder.step()
                optimizer_decoder.step()
                optimizer_property_predictor.step()

                # combine loss for average
                train_reconstruction_loss += float(reconstruction_loss.detach().cpu())
                train_divergence_loss += float(divergence_loss.detach().cpu())
                train_property_loss += float(property_loss.detach().cpu())

                # update progress bar
                t.set_postfix(
                    {'reconstruction_loss': '%.2f' % float(reconstruction_loss),
                     'divergence_loss': '%.2f' % float(divergence_loss),
                     'property_loss': '%.2f' % float(property_loss)}
                )

        # append train loss
        data_train_reconstruction_loss_list_teacher_forcing.append((train_reconstruction_loss / num_batches_train))
        data_train_divergence_loss_list_teacher_forcing.append((train_divergence_loss / num_batches_train))
        data_train_property_loss_list_teacher_forcing.append((train_property_loss / num_batches_train))

        # validation
        tqdm.write("Validation ...")
        with torch.no_grad():
            # predict property of valid data
            vae_encoder.eval()
            vae_decoder.eval()
            property_predictor.eval()

            # calculate validation reconstruction percentage
            valid_count_of_reconstruction = 0
            valid_reconstruction_loss = torch.zeros(size=(1,), dtype=dtype).cpu()
            valid_divergence_loss = torch.zeros(size=(1,), dtype=dtype).cpu()
            valid_property_loss = torch.zeros(size=(1,), dtype=dtype).cpu()

            y_valid_epoch = y_valid.clone()

            # mini-batch training
            pbar = tqdm(range(num_batches_valid), total=num_batches_valid, leave=True)
            with pbar as t:
                for batch_iteration in t:
                    # manual batch iterations
                    start_idx = batch_iteration * batch_size
                    stop_idx = (batch_iteration + 1) * batch_size

                    if batch_iteration == num_batches_valid - 1:
                        stop_idx = len(data_valid)

                    data_valid_batch = MoleculeDataset([data_valid[idx] for idx in range(start_idx, stop_idx)])
                    integer_encoded_valid_batch = integer_encoded_valid[start_idx: stop_idx].to(device)
                    y_valid_batch = y_valid_epoch[start_idx: stop_idx].to(device)

                    # chemprop encode
                    latent_points, mus, log_vars = vae_encoder(data_valid_batch)

                    # property prediction
                    criterion = nn.MSELoss(reduction='none').to(device)
                    predicted_valid_property = property_predictor(latent_points)
                    property_loss = criterion(input=predicted_valid_property, target=y_valid_batch)
                    property_loss = torch.sum(
                        torch.mean(property_loss, dim=0) * property_predictor.normalized_weights_tensor.to(device),
                        dim=0
                    )

                    # decode - teacher forcing
                    reconstruction_loss, acc, predictions, target = vae_decoder(latent_points, integer_encoded_valid_batch)

                    # inference - beam search
                    # debug
                    if (epoch % 10) == 9:  # for every 10 epochs
                        beam_size = 1
                        num_molecules_batch = integer_encoded_valid_batch.shape[0]

                    # try
                    # decoded_predictions_beam = vae_decoder.inference(latent_points, beam_size=beam_size)  # (num_batch * beam_size, sequence length (L))
                    # decoded_predictions_beam = decoded_predictions_beam.cpu().reshape(num_molecules_batch, beam_size, -1)
                    # decoded_molecules = []  # will contain list
                    # for batch_idx in range(num_molecules_batch):
                    #     # answer molecule
                    #     answer_molecule = combine_tokens(tokenids_to_vocab(integer_encoded_valid_batch[batch_idx].tolist(), vae_decoder.vocab, grammar_fragment, include_SOS=False))
                    #     print(answer_molecule)
                    #
                    #     # beam search
                    #     batch_decoded_molecules_beam_set = set()
                    #     for decoded_idx_list in decoded_predictions_beam[batch_idx].tolist():
                    #         decoded_molecule = combine_tokens(tokenids_to_vocab(decoded_idx_list, vae_decoder.vocab, grammar_fragment, include_SOS=True))
                    #         batch_decoded_molecules_beam_set.add(decoded_molecule)
                    #     decoded_molecules.append(batch_decoded_molecules_beam_set)  # append
                    #
                    #     print(batch_decoded_molecules_beam_set)
                    #     # cnt
                    #     if answer_molecule in batch_decoded_molecules_beam_set:
                    #         valid_count_of_reconstruction += 1

                        decoded_predictions = vae_decoder.inference(latent_points, beam_size=beam_size)
                        decoded_molecules = [
                            combine_tokens(tokenids_to_vocab(decoded_predictions[sample][0].tolist(), vae_decoder.vocab, grammar_fragment)) for sample in range(num_molecules_batch)
                        ]
                        answer_molecules = [
                            combine_tokens(tokenids_to_vocab(integer_encoded_valid_batch[sample].tolist(), vae_decoder.vocab, grammar_fragment)) for sample in range(num_molecules_batch)
                        ]

                        # count right & wrong molecules
                        for count_idx in range(num_molecules_batch):
                            decoded_molecule = decoded_molecules[count_idx]
                            # print(decoded_molecule)
                            answer_molecule = answer_molecules[count_idx]
                            if decoded_molecule == answer_molecule:
                                valid_count_of_reconstruction += 1

                    # loss
                    divergence_loss = estimate_maximum_mean_discrepancy(posterior_z_samples=latent_points)

                    # combine loss for average
                    valid_reconstruction_loss += float(reconstruction_loss.detach().cpu())
                    valid_divergence_loss += float(divergence_loss.detach().cpu())
                    valid_property_loss += float(property_loss.detach().cpu())

                    # update progress bar
                    t.set_postfix(
                        {'reconstruction_loss': '%.2f' % float(reconstruction_loss),
                         'divergence_loss': '%.2f' % float(divergence_loss),
                         'property_loss': '%.2f' % float(property_loss)}
                    )

            # append validation loss
            data_valid_reconstruction_loss_list_teacher_forcing.append((valid_reconstruction_loss / num_batches_valid))
            data_valid_divergence_loss_list_teacher_forcing.append((valid_divergence_loss / num_batches_valid))
            data_valid_property_loss_list_teacher_forcing.append((valid_property_loss / num_batches_valid))

            # save encoder, decoder, and property network
            epoch_check_point_dir = Path(os.path.join(check_point_dir, f'epoch_{epoch + 1}'))
            epoch_check_point_dir.mkdir(parents=True, exist_ok=True)
            print(f'Epoch {epoch + 1} check points are saved!')
            torch.save(vae_encoder.state_dict(), os.path.join(epoch_check_point_dir, f'encoder.pth'))
            torch.save(vae_decoder.state_dict(), os.path.join(epoch_check_point_dir, f'decoder.pth'))
            torch.save(property_predictor.state_dict(), os.path.join(epoch_check_point_dir, 'property_predictor.pth'))

            # debug
            # if epoch == num_epochs - 1:  # last epoch
            if (epoch % 10) == 9:
                print(f'Epoch {epoch + 1} valid reconstruction accuracy:')
                print('[VAE] Valid reconstruction (no teacher forcing) accuracy percent rate is %.3f' % (valid_count_of_reconstruction / len(data_valid) * 100), flush=True)

            # check Nan Loss
            return_loss = [
                data_valid_reconstruction_loss_list_teacher_forcing[-1],
                data_valid_divergence_loss_list_teacher_forcing[-1],
                data_valid_property_loss_list_teacher_forcing[-1]
            ]
            flag = 0
            for loss in return_loss:
                if math.isnan(loss):
                    flag = 1

            if flag:
                return np.Inf, np.Inf, np.Inf

            # early stopping
            early_stopping(validation_loss=data_valid_reconstruction_loss_list_teacher_forcing[-1])

            if early_stopping.early_stop:
                tqdm.write(f'===== Early Stopping Is Executed At Epoch {epoch + 1} =====')

                # plot results
                iteration_list = list(range(1, epoch + 2))

                plot_learning_curve(
                    iteration_list=iteration_list,
                    train_loss_teacher_forcing=torch.tensor(
                        data_train_reconstruction_loss_list_teacher_forcing).numpy(),
                    valid_loss_teacher_forcing=torch.tensor(
                        data_valid_reconstruction_loss_list_teacher_forcing).numpy(),
                    save_directory=save_directory,
                    title='Reconstruction Loss'
                )
                plot_learning_curve(
                    iteration_list=iteration_list,
                    train_loss_teacher_forcing=torch.tensor(data_train_divergence_loss_list_teacher_forcing).numpy(),
                    valid_loss_teacher_forcing=torch.tensor(
                        data_valid_divergence_loss_list_teacher_forcing).numpy(),
                    save_directory=save_directory,
                    title='Divergence Loss'
                )
                plot_learning_curve(
                    iteration_list=iteration_list,
                    train_loss_teacher_forcing=torch.tensor(data_train_property_loss_list_teacher_forcing).numpy(),
                    valid_loss_teacher_forcing=torch.tensor(
                        data_valid_property_loss_list_teacher_forcing).numpy(),
                    save_directory=save_directory,
                    title='Property Loss'
                )

                # save encoder, decoder, and property network
                # torch.save(vae_encoder.state_dict(), os.path.join(save_directory, 'encoder.pth'))
                # torch.save(vae_decoder.state_dict(), os.path.join(save_directory, 'decoder.pth'))
                # torch.save(property_predictor.state_dict(), os.path.join(save_directory, 'property_predictor.pth'))

                return data_valid_reconstruction_loss_list_teacher_forcing[-1], \
                data_valid_divergence_loss_list_teacher_forcing[-1], \
                data_valid_property_loss_list_teacher_forcing[-1]

            # train
            vae_encoder.train()
            vae_decoder.train()
            property_predictor.train()

        # step learning rate scheduler
        # if (epoch + 1) % step_size == 0:
        #     encoder_lr_scheduler.step()
        #     decoder_lr_scheduler.step()
        #     property_predictor_lr_scheduler.step()
        #     print('The learning rate is decayed!', flush=True)
        #     print(optimizer_encoder, flush=True)
        #     print(optimizer_decoder, flush=True)
        #     print(optimizer_property_predictor, flush=True)

        # lambda scheduler
        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()
        property_predictor_lr_scheduler.step()

        end = time.time()
        print(f"Epoch {epoch + 1} took {(end - start) / 60.0:.3f} minutes for training", flush=True)

    # return loss
    with torch.no_grad():
        # check NaN loss - duplicated above
        # return_loss = [
        #     data_valid_reconstruction_loss_list_teacher_forcing[-1],
        #     data_valid_divergence_loss_list_teacher_forcing[-1],
        #     data_valid_property_loss_list_teacher_forcing[-1]
        # ]
        # flag = 0
        # for loss in return_loss:
        #     if math.isnan(loss):
        #         flag = 1
        #
        # if flag:
        #     return np.Inf, np.Inf, np.Inf

        print("Plot property prediction ...", flush=True)

        # # eval mode
        # vae_encoder.eval()
        # vae_decoder.eval()
        # property_predictor.eval()
        #
        # # train encode
        # inp_flat_one_hot = data_train.transpose(dim0=1, dim1=2)  # convolution
        # latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)
        # train_prediction = y_scaler.inverse_transform(property_predictor(latent_points).cpu().numpy())
        #
        # # valid encode
        # inp_flat_one_hot = data_valid.transpose(dim0=1, dim1=2)  # convolution
        # latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)
        # valid_prediction = y_scaler.inverse_transform(property_predictor(latent_points).cpu().numpy())
        #
        # # get r2 score & plot
        # y_train_true_original_scale = y_scaler.inverse_transform(y_train.cpu().numpy())
        # y_valid_true_original_scale = y_scaler.inverse_transform(y_valid.cpu().numpy())
        # for idx in range(property_predictor.property_dim):
        #     y_train_true = y_train_true_original_scale[:, idx]
        #     y_valid_true = y_valid_true_original_scale[:, idx]
        #     y_train_prediction = train_prediction[:, idx]
        #     y_valid_prediction = valid_prediction[:, idx]
        #
        #     train_r2_score = r2_score(y_true=y_train_true, y_pred=y_train_prediction)
        #     test_r2_score = r2_score(y_true=y_valid_true, y_pred=y_valid_prediction)
        #
        #     plt.figure(figsize=(6, 6), dpi=300)
        #     plt.plot(y_train_true, y_train_prediction, 'bo',
        #              label='Train R2 score: %.3f' % train_r2_score)
        #     plt.plot(y_valid_true, y_valid_prediction, 'ro',
        #              label='Validation R2 score: %.3f' % test_r2_score)
        #
        #     plt.legend()
        #     plt.xlabel('True')
        #     plt.ylabel('Prediction')
        #
        #     plt.savefig(os.path.join(save_directory, f"property_prediction_{idx}.png"))
        #     plt.show()
        #     plt.close()

    # plot lr
    plt.figure(figsize=(6, 6))
    plt.plot(range(num_epochs), lr_plot)
    plt.xlabel('Epoch (starting from 0)', fontsize=16)
    plt.ylabel('Learning rate', fontsize=16)
    plt.savefig(os.path.join(save_directory, 'lr_plot.png'))
    plt.close()

    # save encoder, decoder, and property network
    # torch.save(vae_encoder.state_dict(), os.path.join(save_directory, 'encoder.pth'))
    # torch.save(vae_decoder.state_dict(), os.path.join(save_directory, 'decoder.pth'))
    # torch.save(property_predictor.state_dict(), os.path.join(save_directory, 'property_predictor.pth'))

    # plot results
    iteration_list = list(range(1, num_epochs + 1))

    plot_learning_curve(
        iteration_list=iteration_list,
        train_loss_teacher_forcing=torch.tensor(data_train_reconstruction_loss_list_teacher_forcing).numpy(),
        valid_loss_teacher_forcing=torch.tensor(data_valid_reconstruction_loss_list_teacher_forcing).numpy(),
        save_directory=save_directory,
        title='Reconstruction Loss'
    )
    plot_learning_curve(
        iteration_list=iteration_list,
        train_loss_teacher_forcing=torch.tensor(data_train_divergence_loss_list_teacher_forcing).numpy(),
        valid_loss_teacher_forcing=torch.tensor(data_valid_divergence_loss_list_teacher_forcing).numpy(),
        save_directory=save_directory,
        title='Divergence Loss'
    )
    plot_learning_curve(
        iteration_list=iteration_list,
        train_loss_teacher_forcing=torch.tensor(data_train_property_loss_list_teacher_forcing).numpy(),
        valid_loss_teacher_forcing=torch.tensor(data_valid_property_loss_list_teacher_forcing).numpy(),
        save_directory=save_directory,
        title='Property Loss'
    )

    return data_valid_reconstruction_loss_list_teacher_forcing[-1], \
        data_valid_divergence_loss_list_teacher_forcing[-1], \
        data_valid_property_loss_list_teacher_forcing[-1]


# def compute_reconstruction_loss(x, x_hat, nop_tensor):  # (num_molecules_batch, max_length, num_vocabs)
#     # get values
#     batch_size = x.shape[0]
#     max_length = x.shape[1]
#
#     inp = x_hat.reshape(-1, x_hat.shape[2])
#     target = x.reshape(-1, x.shape[2]).argmax(1)
#
#     criterion = torch.nn.CrossEntropyLoss(reduction='none')
#     recon_loss = criterion(inp, target)
#     recon_loss = recon_loss.reshape(batch_size, max_length)  # reshape
#
#     # exclude error contribution from nop
#     err_boolean_tensor = torch.ones_like(recon_loss)  # (num_molecules_in_batch, max_length)
#
#     for idx, value in enumerate(nop_tensor):
#         if value >= 0.0:  # exclude
#             err_boolean_tensor[idx, int(value + 1): max_length] = 0.0
#
#     # multiply
#     recon_loss = recon_loss * err_boolean_tensor
#
#     return recon_loss.sum() / err_boolean_tensor.count_nonzero()


def plot_learning_curve(iteration_list,
                        train_loss_teacher_forcing,
                        # train_loss_no_teacher_forcing,
                        valid_loss_teacher_forcing,
                        save_directory,
                        title: str):
    plt.figure(figsize=(8, 8), dpi=300)
    plt.plot(iteration_list, train_loss_teacher_forcing, 'b-', label='Train Loss (teacher forcing)')
    # plt.plot(iteration_list, train_loss_no_teacher_forcing, 'g-', label='Train Loss (no teacher forcing)')
    plt.plot(iteration_list, valid_loss_teacher_forcing, 'r-', label='Validation Loss (teacher forcing)')

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("%s" % title, fontsize=12)

    plt.legend()
    plt.savefig(os.path.join(save_directory, "%s_learning_curve.png" % title))
    plt.show()
    plt.close()


def estimate_maximum_mean_discrepancy(posterior_z_samples: torch.Tensor):
    # set dtype and device
    dtype = posterior_z_samples.dtype
    device = posterior_z_samples.device

    # set dimension and number of samples
    latent_dimension = posterior_z_samples.shape[1]
    number_of_samples = posterior_z_samples.shape[0]

    # get prior samples
    prior_z_samples = torch.randn(size=(number_of_samples, latent_dimension), device=device, dtype=dtype)

    # calculate Maximum Mean Discrepancy with inverse multi-quadratics kernel
    # set value of c - refer to Sec.4 of Wasserstein paper
    c = 2 * latent_dimension * (1.0 ** 2)

    # calculate pp term (p means prior)
    pp = torch.mm(prior_z_samples, prior_z_samples.t())
    pp_diag = pp.diag().unsqueeze(0).expand_as(pp)

    # calculate qq term (q means posterior)
    qq = torch.mm(posterior_z_samples, posterior_z_samples.t())
    qq_diag = qq.diag().unsqueeze(0).expand_as(qq)

    # calculate pq term (q means posterior)
    pq = torch.mm(prior_z_samples, posterior_z_samples.t())

    # calculate kernel
    kernel_pp = torch.mean(c / (c + pp_diag + pp_diag.t() - 2 * pp))
    kernel_qq = torch.mean(c / (c + qq_diag + qq_diag.t() - 2 * qq))
    kernel_pq = torch.mean(c / (c + qq_diag + pp_diag.t() - 2 * pq))

    # estimate mmd
    mmd = kernel_pp + kernel_qq - 2 * kernel_pq

    return mmd


def tokenids_to_vocab(token_ids: list, vocab, grammar_fragment, include_SOS=False):
    """Map token ids back to tokens. Additionally stop generating if encounters a [EOS] token.
    And then convert to SMILES with asterisks.

        Args:
            token_ids: list of token ids
            vocab: vocab as dictionary {str: int}

        Returns:
            list of tokens
    """
    vocab_swap = {v: k for k, v in vocab.items()}
    generated_molecule = []
    if not include_SOS:
        iter_token_ids = token_ids
    else:  # excluding the first [SOS]
        iter_token_ids = token_ids[1:]

    for x in iter_token_ids:
        token = vocab_swap[x]
        if token == '[EOS]':
            break
        generated_molecule.append(token)
    generated_molecule = ''.join(generated_molecule)

    # group selfies
    try:
        decoded_str = Chem.MolToSmiles(grammar_fragment.decoder(generated_molecule))
    except:  # sometimes fails
        return None

    # replace
    p_smi_decoded = Chem.CanonSmiles(decoded_str.replace("I", "*"))

    return p_smi_decoded


def combine_tokens(tokens: list):
    if tokens is None:  # Group SELFIES fails
        return None
    return ''.join(tokens)