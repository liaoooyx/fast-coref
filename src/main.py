import hashlib
import json
import logging
import os
import shutil
import time
from datetime import datetime
from os import path

import hydra
from experiment import Experiment
from omegaconf import OmegaConf

# import wandb


# logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_model_name(config):
    #     masked_copy = OmegaConf.masked_copy(
    #         config, ["datasets", "model", "trainer", "optimizer"]
    #     )
    #     print(f"\nmasked_copy:{masked_copy}\n")
    #     encoded = json.dumps(OmegaConf.to_container(masked_copy), sort_keys=True).encode()
    #     # encoded['seed']=
    #     hash_obj = hashlib.md5()
    #     hash_obj.update(encoded)
    #     hash_obj.update(f"seed: {config.seed}".encode())

    #     model_hash = str(hash_obj.hexdigest())

    currTime = datetime.now().strftime("%y%m%d%H%M%S")

    if len(config.datasets) > 1:
        dataset_name = "joint"
    else:
        dataset_name = list(config.datasets.keys())[0]
        if dataset_name == "litbank":
            cross_val_split = config.datasets[dataset_name].cross_val_split
            dataset_name += f"_cv_{cross_val_split}"

    # model_name = f"{dataset_name}_{model_hash}"
    model_name = f"{dataset_name}_{currTime}"
    return model_name


def main_train(config):
    logger.debug(f"Set up configurartion for training")
    logger.debug(f"Original paths config:{config.paths}")
    # This name will be used to create a path for the {paths.model_dir} in config.yaml
    if config.paths.model_name is None:
        model_name = get_model_name(config)
    else:
        model_name = config.paths.model_name

    # Ignore the original model_dir and create a new path.
    config.paths.model_dir = path.join(config.paths.base_model_dir, config.paths.model_name_prefix + model_name)
    # Ignore the original best_model_dir and create a new path.
    config.paths.best_model_dir = path.join(config.paths.model_dir, "best")

    # Create the folders if not exist.
    for model_dir in [config.paths.model_dir, config.paths.best_model_dir]:
        if not path.exists(model_dir):
            os.makedirs(model_dir)

    # If {paths.model_path} in config.yaml is null, then create/replace both {paths.model_path} and {paths.best_model_dir}
    if config.paths.model_path is None:
        config.paths.model_path = path.abspath(path.join(config.paths.model_dir, config.paths.model_filename))
        config.paths.best_model_path = path.abspath(path.join(config.paths.best_model_dir, config.paths.model_filename))

    # If {paths.model_path} has value and {paths.best_model_dir} is null, then make them be the same path
    # Which means we can't differ the current model and the best model.
    if config.paths.best_model_path is None and (config.paths.model_path is not None):
        config.paths.best_model_path = config.paths.model_path

    # Custom setting
    # We want to fine-tune the pre-trained model. So we copy the pre-trained model from {paths.pretrain_model_dir} to {paths.best_model_dir}
    # When the model is initializing at the training stage, it will load checkpoint from {best_model_path} when {model_dir} is not exist
    # See experiment.py -> def _load_previous_checkpoint(self) for details
    if config.copy_from_pretrained_model and not config.continue_training:
        logger.debug(f"Copying the pretrained model from [{config.paths.pretrain_model_dir}]")
        pretrain_model_path = path.abspath(path.join(config.paths.pretrain_model_dir, config.paths.model_filename))
        if path.exists(config.paths.model_path):
            raise Exception(f"The last checkpoint {config.paths.model_path} exists. The pre-trained model under [{config.paths.pretrain_model_dir}] will not be loaded for fine-tuning.")
        if path.exists(config.paths.best_model_path):
            logger.info(f"A checkpoint exists in {config.paths.best_model_dir} and will be replaced by {pretrain_model_path}")
        if not path.exists(pretrain_model_path):
            raise Exception(f"The pre-trained model [{pretrain_model_path}] is not found.")
        # Copy file
        shutil.copyfile(pretrain_model_path, config.paths.best_model_path)
        logger.debug(f"Copied {pretrain_model_path} -> {config.paths.best_model_path}")
    elif config.continue_training:
        logger.debug(f"Continue training {config.paths.model_path}")

    # Dump config file
    config_file = path.join(config.paths.model_dir, "config.json")
    with open(config_file, "w") as f:
        f.write(json.dumps(OmegaConf.to_container(config), indent=4, sort_keys=True))
    logger.debug(f"Rewrote paths config:{config.paths}")
    return model_name


def main_eval(config):
    logger.debug(f"Set up configurartion for evaluation")
    logger.debug(f"Original paths config:{config.paths}")
    # Must specify paths.model_dir in config.yaml
    if config.paths.model_dir is None:
        raise ValueError("Must specify paths.model_dir in config.yaml when train:False is set")

    # best_model_dir means the best result directly come from the training process.
    # The dir would look like: ./fast-coref/models/coref_litbank_cv_0_fb7c919da4efcfe7579bbdbda9822e4e/best
    best_model_dir = path.join(config.paths.model_dir, "best")
    if path.exists(best_model_dir):
        config.paths.best_model_dir = best_model_dir
    else:
        config.paths.best_model_dir = config.paths.model_dir

    # The path would look like: ./fast-coref/models/coref_litbank_cv_0_fb7c919da4efcfe7579bbdbda9822e4e/best/model.pth
    # or ./fast-coref/models/joint_best/model.pth
    config.paths.best_model_path = path.abspath(path.join(config.paths.best_model_dir, config.paths.model_filename))
    logger.debug(f"Rewrote paths config:{config.paths}")


@hydra.main(config_path="conf", config_name="config")
def main(config):
    # Configuration
    logger.debug(f"Configuration process")
    if config.train:
        model_name = main_train(config)
    else:
        main_eval(config)

        # Identify the model name according to its parent folder name,
        # it would be like: litbank_cv_0_fb7c919da4efcfe7579bbdbda9822e4e, joint_best, etc.
        # Notice that the paths.model_name_prefix in config.yaml (i.e. coref_) will be removed from the folder name.
        # The model name mainly design for wandb which we don't use (and also for showing and logging information).
        model_name = path.basename(path.normpath(config.paths.model_dir))
        # Strip prefix
        if model_name.startswith(config.paths.model_name_prefix):
            model_name = model_name[len(config.paths.model_name_prefix) :]

    # if config.use_wandb:
    #     # Wandb Initialization
    #     try:
    #         wandb.init(
    #             id=model_name,
    #             project="Coreference",
    #             config=dict(config),
    #             resume=True,
    #         )
    #     except:
    #         # Turn off wandb
    #         config.use_wandb = False

    logger.info("Start experiment. Model name: %s", model_name)
    start = time.time()
    Experiment(config)
    end = time.time()
    logger.info("Time: %d minutes", (end - start) / 60)


if __name__ == "__main__":
    import sys

    sys.argv.append(f"hydra.run.dir={path.dirname(path.realpath(__file__))}")
    sys.argv.append("hydra/job_logging=none")
    # Logging level: true - DEBUG, false - INFO
    sys.argv.append("hydra.verbose=true")

    logger.debug(f"Launch fast-coref")
    main()
