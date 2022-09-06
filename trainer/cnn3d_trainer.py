import torch
from torch import nn
from utils.load_model import get_model
from utils.load_dataset import get_dataset
from omegaconf import OmegaConf
from config import *
from models.utils.losses import *

class trainCNN3D(tune.Trainable):

    def setup(self, config):
        # setup function is invoked once training starts
        # setup function is invoked once training starts
        # setup function is invoked once training starts
        self.cfg = OmegaConf.load(config_path + cnn3d_config_file) #here use only vae conf file
        self.model_name = '_'.join((self.cfg.model.name, self.cfg.dataset.name)) + '.h5'

        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']

        self.trainloader, self.valloader, self.weights= get_dataset(self.cfg, batch_size=self.batch_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.resources.gpu_trial else "cpu")

        self.model = get_model(self.cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.criterion = nn.CrossEntropyLoss(weight=self.weights.to(self.device))

    def step(self):
        self.current_ip()
        val_loss = self.trainCNN3D(checkpoint_dir=None)
        result = {"loss": val_loss}
        # if detect_instance_preemption():
        #    result.update(should_checkpoint=True)
        # acc = test(self.models, self.test_loader, self.device)
        # don't call report here!
        return result

    def trainCNN3D(self, checkpoint_dir=None):
        ####Train Loop####
        """
        Set the models to the training mode first and train
        """
        train_loss = []
        patience = 1
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.8, patience=patience
                                                               , threshold=0.0001, threshold_mode='rel',
                                                               cooldown=0, min_lr=9e-8, verbose=True)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for i, (x, y) in enumerate(self.trainloader):

                #f1_score = kwargs["f1_score"]

                x = x.float().to(self.device)
                y = y.float().to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                y_hat = self.model(x).squeeze(-1)
                w = int(y.shape[1] / 2)
                h = int(y.shape[2] / 2)
                central_pixel = y[:, w, h].type(torch.LongTensor).to(self.device)

                # Compute and print loss
                loss = self.criterion(y_hat, central_pixel)
                #f1_score(y_hat.detach().cpu(), central_pixel.detach().cpu())

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
            temp_val_loss = 0.0
            val_steps = 0
            for i, (x, y) in enumerate(self.valloader, 0):
                with torch.no_grad():
                    #f1_score = kwargs["f1_score"]
                    self.model.eval()
                    x = x.float().to(self.device)
                    y = y.float().to(self.device)

                    # forward + backward + optimize
                    y_hat = self.model(x).squeeze(-1)
                    w = int(y.shape[1] / 2)
                    h = int(y.shape[2] / 2)
                    central_pixel = y[:, w, h].type(torch.LongTensor).to(self.device)

                    # Compute and print loss
                    temp_val_loss = self.criterion(y_hat, central_pixel)
                    #f1_score(y_hat.detach().cpu(), central_pixel.detach().cpu())

                    val_steps += 1
            val_loss = temp_val_loss / len(self.valloader)
            val_loss_cpu = val_loss.cpu().item()

            #val_loss_cpu = torch.Tensor(val_loss, torch.LongTensor)
            print("val loss is now", type(val_loss_cpu))
            print('validation_loss {}'.format(val_loss_cpu))
            scheduler.step(val_loss)
            print(type(val_loss_cpu))
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