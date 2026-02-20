import time

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as f

from tqdm import tqdm

from rdkit import Chem

def reconstruct_molecules_group_selfies(encoding_alphabet, one_hot_encoded_vector, grammar_fragment) -> list:
    """
    This function reconstructs a molecule from generated one-hot-encoded vectors using GroupSELFIES.
    :param encoding_alphabet: encoding_alphabet to use
    :param one_hot_encoded_vector: generated one_hot_encoded_vector
    :param grammar_fragment: GroupGrammar to use
    :return: reconstructed_molecules (list)
    """
    # set parameters
    reconstructed_molecules = []

    molecules_num = one_hot_encoded_vector.shape[0]
    reconstruct_one_hot = one_hot_encoded_vector.clone().detach()
    for recon_idx in range(molecules_num):
        gathered_atoms = ''
        one_hot_idx = reconstruct_one_hot[recon_idx]
        for recon_value in one_hot_idx:
            gathered_atoms += encoding_alphabet[recon_value]
        generated_molecule = gathered_atoms.replace('[EOS]', '')  # group selfies
        generated_molecule = '[I]' + generated_molecule  # add the first iodine (i.e., asterisk)

        # decode
        decoded_str = Chem.MolToSmiles(grammar_fragment.decoder(generated_molecule))

        # replace
        canonical_smiles_generated_molecule = Chem.CanonSmiles(decoded_str.replace("I", "*"))
        mol = Chem.MolFromSmiles(canonical_smiles_generated_molecule)
        if mol is None:
            print(f'The mol is not valid: {canonical_smiles_generated_molecule}', flush=True)
            reconstructed_molecules.append(None)
            continue

        # check valid P-smiles
        # (1) the number of '*' should be 2
        asterisk_cnt = 0
        for char in canonical_smiles_generated_molecule:
            if char == '*':
                asterisk_cnt += 1
        if asterisk_cnt != 2:
            print(f'The number of asterisks is {asterisk_cnt}: {canonical_smiles_generated_molecule}')
            reconstructed_molecules.append(None)
            continue

        # (2) '*' should be connected through single bond or terminal diene
        flag = 1
        atoms = mol.GetAtoms()
        bond_order = list()
        for atom in atoms:
            atom_symbol = atom.GetSymbol()
            if atom_symbol == '*':
                if atom.GetDegree() != 1:  # asterisk should be the end of molecule
                    flag = 0  # exclude
                    break
                else:
                    bond_order.append(atom.GetExplicitValence())

        # check bond order
        if bond_order[0] == 1 and bond_order[1] == 1:
            pass
        # elif bond_order[0] == 2 and bond_order[1] == 2:  # terminal diene -> fixed in the new SMART version
        #     pass
        else:
            flag = 0

        # exclude error
        if flag == 0:
            print(f"The connectivity of asterisks is not valid: {canonical_smiles_generated_molecule}", flush=True)
            reconstructed_molecules.append(None)
            continue

        # append
        reconstructed_molecules.append(canonical_smiles_generated_molecule)

    return reconstructed_molecules


def reconstruction(reconstruction_type: str, total_batch_iteration, batch_size, data_tensor, device, dtype,
                   eos_idx, encoding_alphabet, grammar_fragment, vae_encoder, property_predictor, vae_decoder):
    """
    This function performs reconstruction
    :return: property_prediction, z_mean_train if reconstruction_type.lower() == 'train'
        else ->  property_prediction
    """
    tqdm.write(f'{reconstruction_type} Reconstruction ...')
    count_of_reconstruction = 0
    answer_polymer, wrong_prediction_polymer = [], []
    property_prediction = []
    if reconstruction_type.lower() == 'train':
        z_mean_train = []
    pbar = tqdm(range(total_batch_iteration), total=total_batch_iteration, leave=True)
    time.sleep(1.0)
    with pbar as t:
        for batch_iteration in t:
            # manual batch iterations
            start_idx = batch_iteration * batch_size
            stop_idx = (batch_iteration + 1) * batch_size

            if batch_iteration == total_batch_iteration - 1:
                stop_idx = data_tensor.shape[0]

            data_batch = data_tensor[start_idx: stop_idx].to(device)

            # encode
            inp_flat_one_hot = data_batch.transpose(dim0=1, dim1=2)
            latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)

            # store
            if reconstruction_type.lower() == 'train':
                z_mean_train += mus.cpu().tolist()

            # property_prediction
            property_prediction += property_predictor(mus).cpu().tolist()  # TODO - mus?

            # use latent vectors as hidden (not zero-initialized hidden)
            num_molecules_batch = data_batch.shape[0]
            len_max_molec = data_batch.shape[1]
            len_alphabet = data_batch.shape[2]

            # latent points (num_molecules_batch, latent_dim)
            hidden = vae_decoder.init_hidden(latent_points)  # (num_layers, num_molecules_batch, latent_dim). num_layers is doubled in case of bidirectional.
            out_one_hot = torch.zeros_like(data_batch, dtype=dtype, device=device)  # (num_molecules_batch, max_length, num_vocabs)
            x_input = torch.zeros(size=(1, num_molecules_batch, len_alphabet), dtype=dtype, device=device)  # (1, num_molecules_batch, len_alphabet)
            eos_tensor = -torch.ones(size=(num_molecules_batch,), dtype=dtype,  device=device)  # (num_molecules_batch,). placeholder

            # no teacher forcing
            for seq_index in range(len_max_molec):
                out_one_hot_line, hidden = vae_decoder(x=x_input, hidden=hidden)
                out_one_hot[:, seq_index, :] = out_one_hot_line[0]

                # change input - no teacher forcing (one-hot)
                x_input = out_one_hot_line.argmax(dim=-1)
                x_input = f.one_hot(x_input, num_classes=len_alphabet).to(torch.float)

            # find 'eos' idx
            boolean_tensor = out_one_hot.argmax(dim=-1) == eos_idx  # (num_molecules_batch, max_length)
            for molecule_idx in range(num_molecules_batch):
                boolean_tensor_molecule_idx = boolean_tensor[molecule_idx].nonzero().flatten()
                if boolean_tensor_molecule_idx.shape[0] >= 1:  # at least one element exists.
                    eos_tensor[molecule_idx] = boolean_tensor_molecule_idx.min()

            # x_indices
            x_indices = data_batch.argmax(dim=-1)  # (num_molecules_batch, max_length)
            x_hat_prob = f.softmax(out_one_hot, dim=-1)  # (num_molecules_batch, max_length, num_vocabs)
            x_hat_indices = x_hat_prob.argmax(dim=-1)  # (num_molecules_batch, max_length)

            # modify x_hat_indices value to eos_idx after the eos appears
            for idx, value in enumerate(eos_tensor):
                if value >= 0.0:
                    x_hat_indices[idx, int(value): len_max_molec] = eos_idx

            # convert to SMILES string
            answer_molecules = reconstruct_molecules_group_selfies(encoding_alphabet, x_indices, grammar_fragment)
            decoded_molecules = reconstruct_molecules_group_selfies(encoding_alphabet, x_hat_indices, grammar_fragment)
            assert len(answer_molecules) == len(decoded_molecules)

            # count right & wrong molecules
            for count_idx in range(len(answer_molecules)):
                if answer_molecules[count_idx] == decoded_molecules[count_idx]:
                    count_of_reconstruction += 1
                else:
                    answer_polymer.append(answer_molecules[count_idx])
                    wrong_prediction_polymer.append(decoded_molecules[count_idx])

    tqdm.write(f'{reconstruction_type} reconstruction rate is {(100 * (count_of_reconstruction / data_tensor.shape[0])):.3f}%')

    if reconstruction_type.lower() == 'train':
        return property_prediction, z_mean_train
    else:
        return property_prediction


# PCA
def pca_using_common_axis(target_embeddings, pca_eigenvector, mean_embeddings):
    """
    This function performs PCA to the target vectors using eigenvectors obtained for the reference data.
    :param target_embeddings: target embeddings to perform PCA
    :param pca_eigenvector: eigenvectors of a covariance matrix obtained from the reference data.
    :param mean_embeddings: mean embeddings of the reference data.
    :return: pd.Dataframe containing PC components (idx starting from 1)
    """
    # print(pca_eigenvector.shape)  # (number_of_pca_axis=latent_dim, latent_dim)
    latent_dim = pca_eigenvector.shape[1]
    # data = target_embeddings - np.mean(target_embeddings, axis=0) -> not true.
    data = target_embeddings - mean_embeddings  # (number_of_data, latent_dim)
    # cov_matrix = data.T.dot(data) / data.shape[0]
    principal_df = pd.DataFrame(data=None, columns=['PC%d' % (num + 1) for num in range(latent_dim)])
    idx = 0
    for eigenvector in pca_eigenvector:  # pca_eigenvector -> (number_of_pca_axis, latent_dim)
        # get PC coefficient
        idx += 1
        principal_df['PC%d' % idx] = data.dot(eigenvector)

    return principal_df


def latent_space_interpolation(vae_decoder, property_predictor, len_max_molec, eos_idx, encoding_alphabet, grammar_fragment,
                               z_vector_1, z_vector_2, device, dtype, num_interpolations=10):
    """
    This function performs latent space interpolation for decoding.
    :param vae_decoder: The VAE decoder model.
    :param property_predictor: property_predictor
    :param len_max_molec: max number to decode. The actual max length - 1 to exclude the first iodine.
    :param eos_idx: idx to stop decoding
    :param encoding_alphabet: alphabet to use
    :param grammar_fragment: grammar_fragment to use
    :param z_vector_1: The np.array to interpolate
    :param z_vector_2: The np.array to interpolate
    :param device: The device to run the computations on.
    :param dtype: dtype to be used in torch
    :param num_interpolations: The number of interpolation steps.
    :return: A list of decoded interpolated molecules.
    """
    # Perform interpolation
    interpolated_points = np.array([z_vector_1 * (1 - alpha) + z_vector_2 * alpha for alpha in np.linspace(0, 1, num_interpolations)])
    interpolated_points = torch.tensor(interpolated_points, dtype=dtype, device=device)  # tensor

    # property_prediction
    property_prediction = property_predictor(interpolated_points).detach().cpu().numpy()  # TODO - mus?

    # interpolated_points (num_interpolations, latent_dim)
    len_alphabet = len(encoding_alphabet)
    hidden = vae_decoder.init_hidden(interpolated_points)  # (num_layers, num_interpolations, latent_dim). num_layers is doubled in case of bidirectional.
    out_one_hot = torch.zeros((num_interpolations, len_max_molec, len_alphabet), dtype=dtype, device=device)  # (num_interpolations, max_length, num_vocabs)
    x_input = torch.zeros(size=(1, num_interpolations, len_alphabet), dtype=dtype, device=device)  # (1, num_interpolations, len_alphabet)
    eos_tensor = -torch.ones(size=(num_interpolations,), dtype=dtype, device=device)  # (num_molecules_batch,). placeholder

    # no teacher forcing
    for seq_index in range(len_max_molec):
        out_one_hot_line, hidden = vae_decoder(x=x_input, hidden=hidden)
        out_one_hot[:, seq_index, :] = out_one_hot_line[0]

        # change input - no teacher forcing (one-hot)
        x_input = out_one_hot_line.argmax(dim=-1)
        x_input = f.one_hot(x_input, num_classes=len_alphabet).to(torch.float)

    # find 'eos' idx
    boolean_tensor = out_one_hot.argmax(dim=-1) == eos_idx  # (num_molecules_batch, max_length)
    for molecule_idx in range(num_interpolations):
        boolean_tensor_molecule_idx = boolean_tensor[molecule_idx].nonzero().flatten()
        if boolean_tensor_molecule_idx.shape[0] >= 1:  # at least one element exists.
            eos_tensor[molecule_idx] = boolean_tensor_molecule_idx.min()

    # x_indices
    x_hat_prob = f.softmax(out_one_hot, dim=-1)  # (num_molecules_batch, max_length, num_vocabs)
    x_hat_indices = x_hat_prob.argmax(dim=-1)  # (num_molecules_batch, max_length)

    # modify x_hat_indices value to eos_idx after the eos appears
    for idx, value in enumerate(eos_tensor):
        if value >= 0.0:
            x_hat_indices[idx, int(value): len_max_molec] = eos_idx

    # convert to SMILES string
    decoded_molecules = reconstruct_molecules_group_selfies(encoding_alphabet, x_hat_indices, grammar_fragment)

    # idx
    idx_decode_success = [True if molecule else False for molecule in decoded_molecules]

    return decoded_molecules, property_prediction[idx_decode_success]


def decode_from_latent_space(vae_decoder, property_predictor, len_max_molec, eos_idx, z_vectors, encoding_alphabet,
                             grammar_fragment, device, dtype):
    """
    This function performs latent space interpolation for decoding.
    :param vae_decoder: The VAE decoder model.
    :param property_predictor: property_predictor
    :param len_max_molec: max number to decode. The actual max length - 1 to exclude the first iodine.
    :param eos_idx: idx to stop decoding
    :param encoding_alphabet: alphabet to use
    :param grammar_fragment: grammar_fragment to use
    :param device: The device to run the computations on.
    :param dtype: dtype to be used in torch
    :return: A list of decoded interpolated molecules.
    """
    # property_prediction
    num_molecules = z_vectors.shape[0]
    z_vectors_tensor = torch.tensor(z_vectors, dtype=dtype, device=device)  # tensor
    property_prediction = property_predictor(z_vectors_tensor).detach().cpu().numpy()  # TODO - mus?

    # decode
    len_alphabet = len(encoding_alphabet)
    hidden = vae_decoder.init_hidden(z_vectors_tensor)  # (num_layers, num_molecules, latent_dim). num_layers is doubled in case of bidirectional.
    out_one_hot = torch.zeros((num_molecules, len_max_molec, len_alphabet), dtype=dtype, device=device)  # (num_interpolations, max_length, num_vocabs)
    x_input = torch.zeros(size=(1, num_molecules, len_alphabet), dtype=dtype, device=device)  # (1, num_interpolations, len_alphabet)
    eos_tensor = -torch.ones(size=(num_molecules,), dtype=dtype, device=device)  # (num_molecules_batch,). placeholder

    # no teacher forcing
    for seq_index in range(len_max_molec):
        out_one_hot_line, hidden = vae_decoder(x=x_input, hidden=hidden)
        out_one_hot[:, seq_index, :] = out_one_hot_line[0]

        # change input - no teacher forcing (one-hot)
        x_input = out_one_hot_line.argmax(dim=-1)
        x_input = f.one_hot(x_input, num_classes=len_alphabet).to(torch.float)

    # find 'eos' idx
    boolean_tensor = out_one_hot.argmax(dim=-1) == eos_idx  # (num_molecules_batch, max_length)
    for molecule_idx in range(num_molecules):
        boolean_tensor_molecule_idx = boolean_tensor[molecule_idx].nonzero().flatten()
        if boolean_tensor_molecule_idx.shape[0] >= 1:  # at least one element exists.
            eos_tensor[molecule_idx] = boolean_tensor_molecule_idx.min()

    # x_indices
    x_hat_prob = f.softmax(out_one_hot, dim=-1)  # (num_molecules_batch, max_length, num_vocabs)
    x_hat_indices = x_hat_prob.argmax(dim=-1)  # (num_molecules_batch, max_length)

    # modify x_hat_indices value to eos_idx after the eos appears
    for idx, value in enumerate(eos_tensor):
        if value >= 0.0:
            x_hat_indices[idx, int(value): len_max_molec] = eos_idx

    # convert to SMILES string
    decoded_molecules = reconstruct_molecules_group_selfies(encoding_alphabet, x_hat_indices, grammar_fragment)

    # idx
    idx_decode_success = [True if molecule else False for molecule in decoded_molecules]

    return decoded_molecules, property_prediction[idx_decode_success]

def tokenids_to_vocab(token_ids: list, vocab, grammar_fragment, convert_to_smi=False, include_SOS=False):
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

    # compare decoded tokens during training
    if not convert_to_smi:
        return generated_molecule
    else:  # convert to smi during latent space visualization
        try:
            decoded_str = Chem.MolToSmiles(grammar_fragment.decoder(generated_molecule))
        except:  # sometimes fails
            return None

        # replace
        p_smi_decoded = Chem.CanonSmiles(decoded_str.replace("I", "*"))

        return p_smi_decoded
