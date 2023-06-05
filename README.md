# Introducing the softlength (sol) activation function: A practical implementation of Occam's Razor

Artificial Neural Networks (ANNs) traditionally function by accepting input `x`, transforming it through interconnected neural layers, and producing a probability distribution for output y. The uncertainty of `y` is determined by the magnitudes in the ANN's final layer, where larger magnitudes signify less uncertainty and conversely, smaller ones denote more.

This repository introduces a novel method for configuring the probability distribution for `y` using an activation function we have named the Softlength (sol). Our proposed function determines the output's uncertainty based on the 'length', defined as the number of interconnected neural layers used. We reveal that shorter lengths correspond to reduced uncertainty, while longer lengths imply increased uncertainty.

During the training phase, the ANN is conditioned to use fewer neural layers to minimize uncertainty. However, generating the correct result often necessitates a minimum quantity of these layers. This approach is reminiscent of the Occam's razor principle, favoring simpler solutions that employ the fewest possible layers whilst ensuring the results are valid. Thus, this repository presents a practical application of this principle, enhancing efficiency in ANN models without compromising on accuracy.
