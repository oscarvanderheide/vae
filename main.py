"""Main entry point for training VAE models using Hydra."""

import os

import hydra
import pytorch_lightning as pl
import torch
import wandb 
from omegaconf import OmegaConf

from src.config_utils import set_custom_resolvers
from src.visualization import plot_reconstructions


# Set custom resolvers for parsing config files (functions, tuples, etc.)
set_custom_resolvers()


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):
    """Main function for training and evaluating VAE models using Hydra."""

    # For reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # Instantiate Logger
    logger = None
    if "logger" in cfg and cfg.logger is not None:
        # Instantiate the logger first, which initializes wandb run
        logger = hydra.utils.instantiate(cfg.logger) 

    # Create the data module, model, and trainer directly from the config files
    datamodule = hydra.utils.instantiate(cfg.dataset.args)
    model = hydra.utils.instantiate(cfg.model.args)
    trainer = hydra.utils.instantiate(cfg.trainer.args, logger=logger)

    if isinstance(logger, pl.loggers.WandbLogger) and trainer.is_global_zero:
            logger.experiment.config.update( 
                OmegaConf.to_container(cfg, resolve=True),
                allow_val_change=True,
            )

    # Train the model
    trainer.fit(model, datamodule)

    # Visualize reconstruction results
    if cfg.enable_visualization:
        plot_reconstructions(model, datamodule, cfg.output_dir)

    # Save the model
    if cfg.save_model:
        print(f"Saving model to {cfg.output_dir}")
        torch.save(model.state_dict(), os.path.join(cfg.output_dir, "model.pt"))

    # Finish WandB run 
    if isinstance(logger, pl.loggers.WandbLogger):
        wandb.finish()

    return None


if __name__ == "__main__":
    main()
