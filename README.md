# Introducing the softlength activation function: A practical implementation of Occam's razor

Artificial neural networks (ANNs) traditionally function by accepting input `x`, transforming it through interconnected neural layers, and producing a probability distribution for output `y`. The uncertainty of `y` is determined by the magnitudes in the ANN's final layer, where larger magnitudes signify less uncertainty and conversely, smaller ones denote more.

This repository introduces a novel method for configuring the probability distribution for `y` using an activation function we have named the softlength (SoL) function. Our proposed function determines the output's uncertainty based on the 'length', defined as the number of interconnected neural layers used. We reveal that shorter lengths correspond to reduced uncertainty, while longer lengths imply increased uncertainty.

During a typical training phase, an ANN is conditioned to minimize the uncertainty surrounding the correct values of `y`. When using a SoL activation function for `y`, the ANN becomes conditioned to use fewer neural layers to reduce the uncertainty in `y`. This approach is reminiscent of the Occam's razor principle, favoring simpler solutions that employ the fewest possible layers whilst ensuring the results are valid. Thus, this repository presents a practical application of this principle, enhancing efficiency in ANN models without compromising on accuracy.

