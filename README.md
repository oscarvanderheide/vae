## Variational AutoEncoder

This repository contains a Pytorch Lightning-based implementation of the standard Variational AutoEncoder (VAE) 
and several variants. 

The `AbstractVAE(pl.LightningModule)` model implements everything but a `loss_function`. 

 methods needed for a VAE except for the loss function. 

The StandardVAE is a concrete subclass of AbstractVAE that does contain a loss
function consisting of a reconstructio loss and KL divergence term. No other VAE 
variants have been implemented at the moment.

Under the hood, a VAE contains an encoder and a decoder. The encoder first maps
an input sample to a feature vector. The feature vector is then mapped to a 
mean and (log) variance in latent space through fully-connected layers.
The decoder takes a sampled latent vector, maps it to a feature vector with a
fully-connected layer and then generates a sample using a procedure that is 
like the inverse of the feature extractor. The feature extractor and sample generator
together are referred to as the "backbone" of the VAE. Several backbones could be considered: 
CNN, MLP, Transformer and perhaps more.

All methods except for `loss_function` have been defined in the `AbstractVAE(pl.LightningModule)`.