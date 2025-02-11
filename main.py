import argparse
import ray
from ray.tune.schedulers import ASHAScheduler
from omegaconf import OmegaConf
from config import *
#from trainer.vae_trainer import trainVae
from utils.load_trainer import get_trainer

def main(args):

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
    sched = ASHAScheduler(metric="loss", mode="min")
    analysis = tune.run(trainer,
                        scheduler=sched,
                        stop={"training_iteration": 10 ** 16},
                        resources_per_trial={"cpu": 10, "gpu": 1},
                        num_samples=100,
                        checkpoint_at_end=True, #otherwise it fails on multinode?
                        # checkpoint_freq=1,
                        local_dir="~/ray_results",
                        name="{}".format(cfg.model.name),
                        config=config)

    print("Best config is:", analysis.get_best_config(metric="loss"))

if __name__ == "__main__":
    # ip_head and redis_passwords are set by ray cluster shell scripts
    # use the arg parse to call this script from sh script that run the cluster
    # remember to ray start --head on the node you have itneractively
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", help="adress of master")
    parser.add_argument("--password", help="password to connect to master")
    #parser.add_argument("--config_path", default='./train_configurations/', help="echo the string you use here")
    parser.add_argument("--config_file", default='ae', help="the model you want to hpo")
    args = parser.parse_args()

    os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = "12"

    # to test on interactive node
    # first start from the terminal: ray start --head
    # args.address have to be the address of the node
    ray.init(address='auto', _node_ip_address=args.address.split(":")[0], _redis_password=args.password)

    main(args)

