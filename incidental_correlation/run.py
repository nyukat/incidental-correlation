import copy
import os
import pytorch_lightning as pl
import resource
from config import ex
from modules.fiber_module import FIBERTransformerSS
from datamodules.multitask_datamodule import MTDataModule


os.environ["NCCL_DEBUG"] = "INFO"


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)
    model = FIBERTransformerSS(_config)

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode=_config["val_mode"],
        save_last=True,
        filename="best"
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        mode=_config["val_mode"],
        patience=_config["patience"]
    )
    callbacks = [
        checkpoint_callback,
        early_stop_callback
    ]
    logger = pl.loggers.CSVLogger(
        _config["log_dir"],
        name="",
        version=_config["seed"]
    )

    model.freeze()
    task = _config["task"]
    if "vae" in task:
        model.vae.requires_grad_(True)
    elif "multimodal_classify" in task:
        model.multimodal_classifier.requires_grad_(True)
    elif "unimodal_classify" in task:
        model.unimodal_classifier.requires_grad_(True)
    else:
        raise ValueError

    n_accumulate = max(_config["batch_size"] // (_config["per_gpu_batchsize"] * _config["num_gpus"] * _config["num_nodes"]), 1)
    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"],
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=n_accumulate,
        resume_from_checkpoint=_config["resume_from"] # Load everything (model weights, optimizer, lr scheduler, etc)
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
