# Flexible Variational Autoencoder Framework

This repository provides a flexible and extensible framework for training Variational Autoencoders (VAEs) using PyTorch Lightning and Hydra. It's designed to facilitate experimentation with different datasets, model backbones, and VAE architectures, making it a suitable starting point for researchers (particularly in fields like MRI) looking to apply or extend VAE models.

The core strength lies in its modular design, managed by Hydra, allowing users to easily swap components like datasets, backbones, and VAE model types through configuration files or command-line arguments.

The code is available at: [https://github.com/oscarvanderheide/vae](https://github.com/oscarvanderheide/vae)

## Project Structure

-   **`main.py`**: The main script to run training and evaluation, orchestrated by Hydra.
-   **`README.md`**: This file.
-   **`config/`**: Contains Hydra configuration files.
    -   `config.yaml`: Main configuration file defining defaults and experiment structure.
    -   `model/`, `backbone/`, `dataset/`, `trainer/`: Subdirectories holding configurations for different swappable components.
-   **`src/`**: Contains the core source code.
    -   `vae_models/`: Implementations of different VAE models (e.g., `StandardVAE`) inheriting from `AbstractVAE`.
    -   `vae_backbones/`: Implementations of different encoder/decoder architectures (e.g., `Conv`, `MLP`) inheriting from `AbstractBackbone`.
    -   `config_utils.py`: Custom Hydra resolvers or utilities.
    -   `visualization.py`: Code for generating reconstruction plots.
-   **`datasets/`**: Contains PyTorch Lightning DataModules for various datasets, inheriting from `AbstractDataModule`.
-   **`outputs/`**: Default directory where Hydra saves experiment results, logs, and configuration snapshots.
-   **`pyproject.toml` / `uv.lock`**: Project metadata and dependencies for `uv`.

## Core Concepts

-   **`AbstractVAE` (`src/vae_models/abstract_vae.py`)**: A PyTorch Lightning module serving as the base class for all VAE models. It defines the common structure, training steps, and validation logic, leaving the specific `loss_function` to be implemented by subclasses.
-   **`StandardVAE` (`src/vae_models/standard.py`)**: A concrete implementation inheriting from `AbstractVAE`. It uses a standard VAE loss function comprising a reconstruction term and a KL divergence term.
-   **Backbones (`src/vae_backbones/`)**: These define the neural network architecture used for the encoder and decoder parts of the VAE. They inherit from `AbstractBackbone`. Examples include convolutional (`Conv`) and multi-layer perceptron (`MLP`) backbones. The framework is designed so you can easily plug in different backbones.
-   **DataModules (`datasets/`)**: These encapsulate all the steps needed to process data: downloading, cleaning, transforming, and loading. They are based on PyTorch Lightning's `LightningDataModule` and inherit from `AbstractDataModule`. This allows for easy switching between datasets like MNIST, FashionMNIST, or BraTS.

## Getting Started

### Environment Setup

It is recommended to use a virtual environment. This project uses `uv` for fast dependency management.

1.  **Install `uv`**: Follow the instructions at [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv).
2.  **Clone the Repository**:
    ```bash
    git clone https://github.com/oscarvanderheide/vae.git
    cd vae
    ```
3.  **Run**: The `uv run` command (used below) will automatically create a virtual environment and install dependencies based on `pyproject.toml` / `uv.lock` if they are not already present in an active environment.

### Running Experiments

Experiments are run using `uv run main.py`. Hydra manages the configuration.

-   **Run with default configuration:**
    (Uses settings defined in `config/config.yaml`, e.g., `standard_vae`, `conv` backbone, `fashion_mnist` dataset)
    ```bash
    uv run main.py
    ```

-   **Override configuration via command line:**
    Hydra's command-line interface is powerful for experimentation.

    *   **Selecting Components:** Choose different configuration files for model, backbone, or dataset.
        ```bash
        # Run with MNIST dataset and MLP backbone
        uv run main.py dataset=mnist backbone=mlp

        # Run with the BraTS dataset (ensure data is available)
        uv run main.py dataset=brats backbone=conv # Adjust backbone/model as needed
        ```

    *   **Overriding Specific Parameters:** Change individual parameters within a configuration group using dot notation.
        ```bash
        # Change the learning rate (assuming it's in model.args.optimizer.lr)
        uv run main.py model.args.optimizer.lr=0.0005

        # Change the number of epochs in the trainer
        uv run main.py trainer.max_epochs=50

        # Change a parameter within the chosen backbone (e.g., number of channels in conv)
        uv run main.py backbone=conv backbone.args.channels=[16,32,64]
        ```

    *   **Parameter Sweeps (Multirun):** Run multiple experiments by specifying comma-separated values or ranges. Use the `-m` flag for multirun.
        ```bash
        # Sweep over different learning rates
        uv run main.py -m model.args.optimizer.lr=0.001,0.0005,0.0001

        # Sweep over datasets and backbones
        uv run main.py -m dataset=mnist,fashion_mnist backbone=mlp,conv

        # Sweep over a range of seeds
        uv run main.py -m seed=range(42,45)
        ```

-   **Output:** Results, logs, and configuration snapshots for each run are saved in the `outputs/` directory (or the directory specified in `hydra.run.dir`), organized by date and time.

## Configuration with Hydra

Hydra allows for flexible configuration management.

-   **`config/config.yaml`**: Sets the default components (`model`, `backbone`, `dataset`, `trainer`) and global parameters (`seed`, `experiment_name`).
-   **Component Configurations**: Files within `config/model/`, `config/backbone/`, `config/dataset/`, and `config/trainer/` define the parameters for each specific component.
-   **`_target_` Key**: Inside component YAML files (e.g., `config/dataset/mnist.yaml`), the `_target_` key specifies the Python class to instantiate (e.g., `datasets.mnist.MNISTDataModule`). Hydra uses this along with the other keys as arguments to the class constructor.
    ```yaml
    # Example: config/dataset/mnist.yaml
    name: mnist
    args:
      _target_: datasets.mnist.MNISTDataModule
      data_dir: ${oc.env:DATA_DIR}/mnist # Example using an environment variable
      batch_size: 64
      num_workers: 4
    ```
-   **Custom Resolvers (`src/config_utils.py`)**: This file can define custom functions accessible within Hydra configurations (e.g., for complex parameter calculations).

## Extending the Framework

Adding new components is straightforward:

1.  **New Dataset:**
    -   Create a Python class inheriting from `datasets.AbstractDataModule`.
    -   Implement the necessary methods (`prepare_data`, `setup`, `train_dataloader`, etc.).
    -   Create a corresponding YAML configuration file in `config/dataset/` specifying the `_target_` as your new class and any necessary arguments.

2.  **New Backbone:**
    -   Create a Python class inheriting from `src.vae_backbones.AbstractBackbone`.
    -   Implement the encoder and decoder logic (`forward` method returning features, `decode` method generating output). Ensure it's compatible with the VAE model structure.
    -   Create a corresponding YAML configuration file in `config/backbone/` specifying the `_target_` and arguments.

3.  **New VAE Model:**
    -   Create a Python class inheriting from `src.vae_models.AbstractVAE`.
    -   Implement the `loss_function` method. You might also need to override other methods like `forward` or `step` depending on your VAE variant.
    -   Create a corresponding YAML configuration file in `config/model/` specifying the `_target_` and arguments (which likely include instantiating a backbone via its config).

## Visualization

The framework includes basic visualization of model reconstructions.

-   Controlled by the `enable_visualization` flag in `config/config.yaml` (or overridden via command line: `uv run main.py enable_visualization=true/false`).
-   When enabled, plots comparing original and reconstructed images from the validation set are saved in the experiment's output directory.
-   The implementation is in `src/visualization.py`.