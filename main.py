import argparse
import ray
from ray.tune.schedulers import ASHAScheduler
from omegaconf import OmegaConf
from config import *
from utils.load_trainer import get_trainer
from datetime import datetime

def main(args):
    now = datetime.now()
    date = now.strftime("%D:%H:%M:%S")
    print(date)

    print(args.address, args.password)
    #config_path = os.path.join(os.getcwd(), rel_conf_path)
    cfg = OmegaConf.load(config_path + args.config_file + '.yaml')

    config = {}
    for k, v in cfg.tune_config.items():
        try:
            config[k] = ray_mapper[v.split('(')[0]]([float(s) if '.' in s else int(s) for s in v.split(v.split('(')[0])[1].\
                                                strip("()").strip("[]").split(',')])
            print([float(s) if '.' in s else int(s) for s in v.split(v.split('(')[0])[1].\
                                                strip("()").strip("[]").split(',')])
        except:
            config[k] = ray_mapper[v.split('(')[0]]([s.strip(' ').strip("''") for s in v.split(v.split('(')[0])[1]\
                                                    .strip("()").strip("[]").split(',')])
            print([s.strip(' ').strip("''") for s in v.split(v.split('(')[0])[1].strip("()")\
                  .strip("[]").split(',')])

    trainer = get_trainer(cfg)
    if cfg.resources.gpu_trial != 0:
        resources_per_trial = {"cpu":cfg.resources.cpu_trial, "gpu": cfg.resources.gpu_trial}
    else:
        resources_per_trial = {"cpu": cfg.resources.cpu_trial}


    sched = ASHAScheduler(metric=cfg.opt.tune_report, mode="min", max_t=400)
    analysis = tune.run(trainer,
                        scheduler=sched,
                        stop={"training_iteration": 10 ** 16},
                        resources_per_trial=resources_per_trial,
                        num_samples=200,
                        checkpoint_at_end=True, #otherwise it fails on multinode?
                        #checkpoint_freq=1,
                        local_dir="~/ray_results",
                        name="{}/{}".format(cfg.model.name, date.replace('/', '-')),
                        config=config)

    print("Best config is:", analysis.get_best_config(metric="val_loss", mode="min"))

    #print("Best trial final validation loss: {}".format(
    #    best_trial.last_result["loss"]))
    #print("Best trial final validation accuracy: {}".format(
    #    best_trial.last_result["accuracy"]))

    #best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    #device = "cpu"
    #if torch.cuda.is_available():
    #    device = "cuda:0"
    #    if gpus_per_trial > 1:
    #        best_trained_model = nn.DataParallel(best_trained_model)
    #best_trained_model.to(device)

    #best_checkpoint_dir = best_trial.checkpoint.value
    #model_state, optimizer_state = torch.load(os.path.join(
    #    best_checkpoint_dir, "checkpoint"))
    #best_trained_model.load_state_dict(model_state)

    #test_acc = test_accuracy(best_trained_model, device)
    #print("Best trial test set accuracy: {}".format(test_acc))

if __name__ == "__main__":
    # ip_head and redis_passwords are set by ray cluster shell scripts
    # use the arg parse to call this script from sh script that run the cluster
    # remember to ray start --head on the node you have itneractively
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", help="adress of master")
    parser.add_argument("--password", help="password to connect to master")
    #parser.add_argument("--config_path", default='./train_configurations/', help="echo the string you use here")
    parser.add_argument("--config_file", default='cnn3d', help="the model you want to hpo")
    args = parser.parse_args()

    os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = "12"

    # to test on interactive node
    # first start from the terminal: ray start --head
    # args.address have to be the address of the node otherwis uncomment ray.init(address='auto') line
    ray.init(address='auto')
    #ray.init(address='auto', _node_ip_address=args.address.split(":")[0], _redis_password=args.password)

    main(args)

