import os
import numpy as np
import torch
from torch import nn
from ray import tune
from utils.load_model import get_model
from utils.load_dataset import get_dataset
from omegaconf import OmegaConf
from config import *
#from models.utils.losses import *

class trainCNN3D(tune.Trainable):

    def setup(self, config):
        # setup function is invoked once training starts
        # setup function is invoked once training starts
        # setup function is invoked once training starts
        #print(config_path)
        self.cfg = OmegaConf.load(config_path + cnn3d_config_file) #here use only vae conf file
        self.model_name = '_'.join((self.cfg.model.name, self.cfg.dataset.name)) + '.h5'
        self.class_number = self.cfg.model.class_number

        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.patience = config['lr_patience']
        self.num_filter = config['num_filter']
        self.act = config['act']
        self.filter_size = config['filter_size']
        self.patch_size = config['patch_size']

        self.best_val_loss = 10**16

        self.cfg['model']['num_filter'] = self.num_filter
        self.cfg['model']['filter_size'] = self.filter_size
        self.cfg['model']['act'] = self.act
        self.cfg['dataset']['patch_size'] = self.patch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.resources.gpu_trial else "cpu")

        if self.cfg.opt.k_fold_cv:
            self.trainloader_list, self.valloader_list, self.weights_list, self.metrics = get_dataset(self.cfg, batch_size=self.batch_size,
                                                                            patch_size=self.patch_size, from_dictionary=self.cfg.dataset.from_dictionary)
            self.model_list = [get_model(self.cfg, num_filter=self.num_filter, act=self.act, filter_size=self.filter_size).to(
                self.device) for i in range(self.cfg.opt.k_fold_cv)]
            self.optimizer_list = [torch.optim.Adam(model.parameters(), lr=self.lr) for model in self.model_list]
            self.scheduler_list = [torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_list[i], 'min', factor=0.8,
                                                                        patience=self.patience
                                                                        , threshold=0.0001, threshold_mode='rel',
                                                                        cooldown=0, min_lr=9e-8, verbose=True) for i in range(len(self.model_list))]
            if self.class_number > 1:
                self.criterion_list = [nn.CrossEntropyLoss(weight=self.weights_list[i]).to(self.device) for i in range(len(self.model_list))]
            else:
                self.criterion_list =  [nn.BCEWithLogitsLoss(pos_weight=self.weights_list[i]).to(self.device) for i in range(len(self.model_list))]

            if "accuracy" in self.metrics:
                self.acc = self.model_list[0].acc.to(self.device)
            if "f1_score" in self.metrics:
                self.f1_score = self.model_list[0].f1_score.to(self.device)

        else:
            self.trainloader, self.valloader, self.testloader, self.weights, self.metrics = get_dataset(self.cfg, batch_size=self.batch_size,
                                                                            patch_size=self.patch_size, from_dictionary=self.cfg.dataset.from_dictionary)
            self.model = get_model(self.cfg, num_filter=self.num_filter, act=self.act,  filter_size=self.filter_size).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.8,
                                                                        patience=self.patience
                                                                        , threshold=0.0001, threshold_mode='rel',
                                                                        cooldown=0, min_lr=9e-8, verbose=True)

            if self.class_number > 1:
                self.criterion = nn.CrossEntropyLoss(weight=self.weights).to(self.device)
            else:
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.weights).to(self.device)

            if "accuracy" in self.metrics:
                self.acc = self.model.acc.to(self.device)
            if "f1_score" in self.metrics:
                self.f1_score = self.model.f1_score.to(self.device)


 #use sigmoid under the hood >>> 1 neuron (1 class)

        # BCEWithLogitLoss and BCE wants 1 neuron
        # one neuron and no act in the net >>> the weight lenght is equal to the number of classes (if binary >>> lenght = 1 if n=3 >>> lenght =3)
        # you can also use 2 neurons but then the Metric doens' works fine anymore
        # linear(self.n_classes -1)
        # accurary()
        # F1 score()
        # requires to switch to float in the criterion loss
        # F1 Score works fine and different from Accuracy becuase we don't have a vector thar has C dinension in output like for crossentropy
        # + weight + Metrics, - adapt to multiclass

        # BCE
        # one neuron and activation (sigmoid) needed >>> the weight should be a vector with the weifght of each example
        # linear(1)
        # accurary()
        # F1 score()
        # F1 Score works fine and different from Accuracy becuase we don't have a vector thar has C dinension in output like for crossentropy
        # - weight + Metrics, - adapt to multiclass

        #Categorical Crossentropy wants 2 neurons
        # self.criterion = nn.CrossEntropyLoss(weight=self.weights).to(self.device)
        # it needs two neuron and have already softmax implemente
        # linear(2)
        # accurary(2)
        # F1 score(2)
        # F1 Score schould receive a vector (n,c) in which you have 1 in place of the correct labe: [0,1][1,0]
        # weight are of lenght C
        # + weight - Metrics, - adapt to multiclass


    def step(self):
        self.current_ip()

        result = self.trainCNN3D(checkpoint_dir=None)
        #test_results = self.testCNN3D(checkpoint_dir=None)
        #result.update(test_results)
        print('these are the results dict',result)
        #result = {"val_loss": val_loss, "test_loss": test_loss}

        #if detect_instance_preemption():
        #    result.update(should_checkpoint=True)
        # acc = test(self.models, self.test_loader, self.device)
        # don't call report here!
        return result

    def trainCNN3D(self, checkpoint_dir=None):
        ####Train Loop####
        """
        Set the models to the training mode first and train
        """
        # each step should be last all the epoch times if you don't return after validation step
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            #model.apply(reset_weights)
            fold_overall_train_loss = []
            fold_overall_val_loss = []
            fold_overall_val_f1 = []
            fold_overall_val_acc = []
            for fold, (train_dl, val_dl) in enumerate(zip(self.trainloader_list, self.valloader_list)):
                temp_train_loss = 0.0
                train_steps = 0
                self.model_list[fold].train()
                for i, (x, y) in enumerate(train_dl):

                    x = x.float().to(self.device)
                    y = y.float().to(self.device)

                    # zero the parameter gradients
                    self.optimizer_list[fold].zero_grad()

                    # forward + backward + optimize
                    model = self.model_list[fold]
                    y_hat = model(x).squeeze(-1)
                    w = int(y.shape[1] / 2)
                    h = int(y.shape[2] / 2)
                    central_pixel = y[:, w, h].type(torch.LongTensor).to(self.device)

                    # Compute and print loss
                    loss = self.criterion_list[fold](y_hat, central_pixel.float())
                    loss.backward()
                    optimizer = self.optimizer_list[fold]
                    optimizer.step()

                    temp_train_loss += loss
                    train_steps += 1
                    if i % 10 == 0:
                        print(" training Loss: {} on fold {}".format(loss.item(), fold))

                train_loss = temp_train_loss / train_steps
                self.train_loss_cpu = train_loss.cpu().item()
                fold_overall_train_loss.append(self.train_loss_cpu)  # buffer for each fold train loss at the end of the epoch
                ###############################################
                # eval mode for evaluation on validation dataset
                ###############################################
                # Validation loss
                temp_val_loss = 0.0
                val_steps = 0
                y_hat_tensor = torch.zeros_like(y_hat)
                central_pixel_tensor = torch.zeros_like(central_pixel)
                model.eval()
                for i, (x, y) in enumerate(val_dl, 0):
                    with torch.no_grad():
                        x = x.float().to(self.device)
                        y = y.float().to(self.device)

                        # forward + backward + optimize
                        y_hat = model(x).squeeze(-1)

                        w = int(y.shape[1] / 2)
                        h = int(y.shape[2] / 2)
                        central_pixel = y[:, w, h].type(torch.LongTensor).to(self.device)

                        if i == 0:
                            y_hat_tensor = torch.zeros_like(y_hat)
                            central_pixel_tensor = torch.zeros_like(central_pixel)
                        else:
                            y_hat_tensor = torch.cat((y_hat_tensor.cpu(), y_hat.detach().cpu()), 0)
                            central_pixel_tensor = torch.cat((central_pixel_tensor.cpu(), central_pixel.detach().cpu()), 0)

                        # Compute and print loss
                        temp_val_loss += self.criterion_list[fold](y_hat, central_pixel.float())

                        y_hat_tensor = torch.cat((y_hat_tensor.cpu(), y_hat.detach().cpu()), 0)
                        central_pixel_tensor = torch.cat((central_pixel_tensor.cpu(), central_pixel.detach().cpu()), 0)
                        val_steps += 1

                val_loss = temp_val_loss / val_steps
                self.val_loss_cpu = val_loss.cpu().item()
                fold_overall_val_loss.append(self.val_loss_cpu)

                scheduler = self.scheduler_list[fold]
                try:
                    f1_score = self.f1_score(y_hat_tensor.to(self.device), central_pixel_tensor.to(self.device)).cpu().item()
                    acc = self.acc(y_hat_tensor.to(self.device), central_pixel_tensor.to(self.device)).cpu().item()
                    print("val Loss: {} and f1_score {} for fold {}".format(self.val_loss_cpu , f1_score, fold))
                    scheduler.step(val_loss)
                    fold_overall_val_acc.append(acc)
                    fold_overall_val_f1.append(f1_score)
                    metrics_flag=True

                except:
                    print('validation_loss {} in fold {}'.format(self.val_loss_cpu, fold))
                    metrics_flag = False
                    scheduler.step(val_loss)
                    if self.val_loss_cpu < self.best_val_loss:
                        self.best_val_loss = self.val_loss_cpu
                        return {"train_loss": self.train_loss_cpu,
                                "val_loss":self.val_loss_cpu, "should_checkpoint": True}
                    else:
                        return {"train_loss": self.train_loss_cpu, "val_loss":self.val_loss_cpu}

                self.model_list[fold] = model
                self.optimizer_list[fold] = optimizer
                self.scheduler_list[fold] = scheduler

            if metrics_flag:
                overall_train_loss = np.mean(fold_overall_train_loss)
                overall_val_loss = np.mean(fold_overall_val_loss)
                overall_f1_score = np.mean(fold_overall_val_loss)
                overall_acc = np.mean(fold_overall_val_acc)
                print("val Loss: {} and f1_score {} for overall fold".format(overall_val_loss, overall_f1_score))
                if overall_val_loss < self.best_val_loss:
                    self.best_val_loss = overall_val_loss
                    return {"train_loss": overall_train_loss, "val_loss": overall_val_loss, "val_acc":overall_acc ,
                            "val_f1": overall_f1_score, "should_checkpoint": True}
                else:
                    overall_train_loss = np.mean(fold_overall_train_loss)
                    overall_val_loss = np.mean(fold_overall_val_loss)
                    return {"train_loss": overall_train_loss,
                            "val_loss": overall_val_loss, "val_acc": overall_acc, "val_f1": overall_f1_score}

            else:
                overall_train_loss = np.mean(fold_overall_train_loss)
                overall_val_loss = np.mean(fold_overall_val_loss)
                print('validation_loss {} in for overall'.format(self.val_loss_cpu, fold))
                self.scheduler_list[fold].step(val_loss)
                if self.val_loss_cpu < self.best_val_loss:
                    self.best_val_loss = self.val_loss_cpu
                    return {"train_loss": overall_train_loss, "val_loss": overall_val_loss,
                             "should_checkpoint": True}
                else:
                    return {"train_loss": overall_train_loss, "val_loss": overall_val_loss}


    def testCNN3D(self, checkpoint_dir=None):
        ###############################################
        # to implement
        ###############################################
        raise NotImplementedError



    def save_checkpoint(self, checkpoint_dir):
        print("this is the checkpoint dir {}".format(checkpoint_dir))
        checkpoint_path_list = []
        for fold, model in enumerate(self.model_list):
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizer_list[fold].state_dict(),
                'loss': self.val_loss_cpu,
                'cfg': self.cfg
            }, f"{checkpoint_dir}/model_{fold}.pt")
            #checkpoint_path_list.append(os.path.join(checkpoint_dir, "model_{}.pt".format(fold)))
        #torch.save(self.model.state_dict(), checkpoint_path)
        return os.path.join(checkpoint_dir, "model.pt".format(fold))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        #self.model.load_state_dict(torch.load(checkpoint_path))
        # this is currently needed to handle Cori GPU multiple interfaces

    def current_ip(self):
        import socket
        hostname = socket.getfqdn(socket.gethostname())
        self._local_ip = socket.gethostbyname(hostname)
        return self._local_ip