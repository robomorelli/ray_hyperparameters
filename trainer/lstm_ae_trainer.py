import os

import torch.nn
from ray import tune
from utils.load_model import get_model
from utils.load_dataset import get_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from config import *
from models.utils.losses import *

class trainLSTMAe(tune.Trainable):

    def setup(self, config):
        self.cfg = OmegaConf.load(config_path + lstm_ae_config_file) #here use only vae conf file
        self.model_name = '_'.join((self.cfg.model.name, self.cfg.dataset.name)) + '.h5'

        #following config keys have to be in the config file of this model
        self.seq_in_length = config['seq_in_length']
        self.n_features = config['n_features']
        self.embedding_dim = config['embedding_dim']
        self.latent_dim = config['latent_dim']
        self.n_layers = config['n_layers']
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.patience = config['lr_patience']

        # to write on cfg to have later on load dataset
        self.cfg.dataset.sequence_length = self.seq_in
        self.cfg.out_window = self.seq_in

        self.trainloader, self.valloader, _, _\
            , self.original_dim = get_dataset(self.cfg, batch_size=self.batch_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.resources.gpu_trial else "cpu")
        self.model = get_model(self.cfg, seq_in_length=self.seq_in_length, n_features=self.n_features, embedding_dim=self.embedding_dim,
                                latent_dim=self.latent_dim, n_layers=self.n_layers, seq_out_lenght=self.seq_in_length).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=0.001, gamma=0.5)

        self.best_val_loss = 10 ** 16

    def step(self):
        self.current_ip()
        val_loss = self.train_ae(checkpoint_dir=None)
        result = {"val_loss": val_loss}
        return result

    def train_lstm_ae(self, checkpoint_dir=None):
        ####Train Loop####
        """
        Set the models to the training mode first and train
        """
        for epoch in tqdm(range(self.epochs), unit='epoch'):
            train_loss = 0
            train_steps = 0
            for i, batch in tqdm(enumerate(self.trainloader), total=len(self.trainloader), unit="batch"):
                self.model.train()
                self.optimizer.zero_grad()

                # y.requires_grad_(True)
                x, enc, y_o = self.model(batch[0].to(self.device))
                loss = self.criterion(y_o.to(self.device), batch[1].to(self.device))
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                self.train_loss += loss.item()
                self.train_steps += 1

                # if (i + 1) % config['gradient_accumulation_steps'] == 0:
                self.optimizer.step()
                self.scheduler.step()

                if i % 10 == 0:
                    print("Loss:")
                    print(loss.item())

            print('train loss at the end of epoch is ', self.train_loss / self.train_steps)

            self.model.eval()
            val_steps = 0
            temp_val_loss = 0
            with torch.no_grad():
                for i, batch in tqdm(enumerate(self.valloader), total=len(self.valloader), desc="Evaluating"):
                    with torch.no_grad():
                        x_o, enc, y_o = self.model(batch[0].to(self.device))
                        loss = self.criterion(x_o.to(self.device), y_o.to(self.device)).item()
                        temp_val_loss += loss
                        val_steps += 1

                temp_val_loss = temp_val_loss / val_steps
                print('eval loss {}'.format(temp_val_loss))
                if self.val_loss_cpu < self.best_val_loss:
                    self.best_val_loss = self.val_loss_cpu
                    return {"train_loss": self.train_loss_cpu,
                            "val_loss": self.val_loss_cpu, "should_checkpoint": True}


        train_loss = []
        patience = 1
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.8, patience=patience
                                                               , threshold=0.0001, threshold_mode='rel',
                                                               cooldown=0, min_lr=9e-8, verbose=True)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for i, (x, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()

                x = x.to(self.device)
                y = y.to(self.device)
                out = self.model(x)

                loss = torch.nn.BCELoss()(out, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                train_loss.append(loss.item())

                if i % 10 == 0:
                    print("Loss: {}".format(loss.item()))

            ###############################################
            # eval mode for evaluation on validation dataset
            ###############################################

            # Validation loss
            # val_loss = 0.0
            temp_val_loss = 0.0
            val_steps = 0
            self.model.eval()
            for i, (x, y) in enumerate(self.valloader, 0):
                with torch.no_grad():
                    x = x.to(self.device)

                    self.optimizer.zero_grad()

                    x = x.to(self.device)
                    y = y.to(self.device)
                    out = self.model(x)

                    temp_val_loss += torch.nn.BCELoss()(out, y)

                    val_steps += 1
            val_loss = temp_val_loss / len(self.valloader)
            val_loss_cpu = val_loss.cpu().item()
            print('validation_loss {}'.format(val_loss_cpu))
            scheduler.step(val_loss)

            return val_loss_cpu

    def save_checkpoint(self, checkpoint_dir):
        print("this is the checkpoint dir {}".format(checkpoint_dir))
        checkpoint_path = os.path.join(checkpoint_dir, self.model_name)
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))

        # this is currently needed to handle Cori GPU multiple interfaces

    def current_ip(self):
        import socket
        hostname = socket.getfqdn(socket.gethostname())
        self._local_ip = socket.gethostbyname(hostname)
        return self._local_ip