import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, chemprop_mpn, in_dimension, layer_1d, layer_2d, layer_3d,
                 latent_dimension):
        """
        Fully Connected layers to encode molecule to latent space
        """
        super(Encoder, self).__init__()
        self.chemprop_mpn = chemprop_mpn
        self.latent_dimension = latent_dimension

        # Reduce dimension up to second last layer of Encoder
        self.encode_nn = nn.Sequential(
            nn.Linear(in_dimension, layer_1d),
            nn.BatchNorm1d(layer_1d),
            nn.LeakyReLU(),
            nn.Linear(layer_1d, layer_2d),
            nn.BatchNorm1d(layer_2d),
            nn.LeakyReLU(),
            nn.Linear(layer_2d, layer_3d),
            nn.BatchNorm1d(layer_3d),
            nn.LeakyReLU()
        )

        # Latent space mean
        self.encode_mu = nn.Linear(layer_3d, latent_dimension)

        # Latent space variance
        self.encode_log_var = nn.Linear(layer_3d, latent_dimension)

    @staticmethod
    def reparameterize(mu, log_var):
        """
        This trick is explained well here:
            https://stats.stackexchange.com/a/16338
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, batch):
        """
        Pass throught the Encoder.
        x: MoleculeDataset (batch)
        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        """
        # get data
        mol_batch, features_batch, target_batch, atom_descriptors_batch, atom_features_batch, bond_descriptors_batch, bond_features_batch, constraints_batch, data_weights_batch = \
            batch.batch_graph(), batch.features(), batch.targets(), batch.atom_descriptors(), \
                batch.atom_features(), batch.bond_descriptors(), batch.bond_features(), batch.constraints(), batch.data_weights()

        # Get ChemProp MPN fingerprints
        chemprop_mpn_fingerprints = self.chemprop_mpn.fingerprint(
            mol_batch,
            features_batch,
            atom_descriptors_batch,
            atom_features_batch,
            bond_descriptors_batch,
            bond_features_batch,
            fingerprint_type='MPN',
        )

        # Get results of encoder network
        h1 = self.encode_nn(chemprop_mpn_fingerprints)

        # latent space
        mu = self.encode_mu(h1)
        log_var = self.encode_log_var(h1)

        # Reparameterize
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
