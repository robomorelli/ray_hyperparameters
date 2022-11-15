from models.vae import VAE
from models.autoencoder import AE
from models.lstm_ae import LSTM_AE
from models.conv_ae import CONV_AE
from models.cnn3d import CNN3D

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

    if cfg.model.name == "lstm_ae":
        model = LSTM_AE(cfg, **kwargs)
        return model

    if cfg.model.name == "conv_ae":
        model = CONV_AE(cfg, **kwargs)
        return model

    if cfg.model.name == "cnn3d":
        model = CNN3D(cfg, **kwargs)
        # optimizer and criterion should be defined here
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # criterion = nn.CrossEntropyLoss(weight=self.weights.to(self.device))
        return model
