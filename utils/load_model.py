from models.vae import VAE
from models.autoencoder import AE

def get_model(cfg, **kwargs):
    """
    Get the dataset.
    :param cfg:  configuration file
    :return: model
    """
    if cfg.model.name == "vae":
        model = VAE(cfg, original_dim=kwargs['original_dim'], intermediate_dim=kwargs['intermediate_dim'],
                    latent_dim=kwargs['latent_dim'],
                    n_lognorm=kwargs['n_lognorm'], n_binomial=kwargs['n_binomial'])
        return model

    if cfg.model.name == "ae":
        model = AE(cfg, original_dim=kwargs['original_dim'], intermediate_dim=kwargs['intermediate_dim'],
                    code_dim=kwargs['code_dim'])
        return model
