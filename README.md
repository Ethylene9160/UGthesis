# UGthesis

## Biological Models

the [bio_models.ipynb](bio_models.ipynb) shows the responses of IZH, MIZH, LIF and HH model. You do not need to do any customization.

## Spiking Neurons for Classical Hopfield Network

In [spiking_network.ipynb](spiking_network.ipynb), I provide the response for pattern-retrival in classical hopfield with different biological models, including the retrival result, the oscillation of the network and the memberance track of a single neuron model.

* Customization: You could change the parameter value of variable `param` in section **Parameter**. I provide 4 models for this parameter: `izh_param`, `mizh_param`, `lif_param` and `hh_param`.

## STDP for Graph Retrival

The [stdp_graph.ipynb](stdp_graph.ipynb) compares the retrival results with Hebbian Learning rule and STDP rule in threshold complex hopfield network. You do not need to do customization

## STDP for Biological Hopfield Networks: Oscillation Retrival

In [stdp_spiking.ipynb](stdp_spiking.ipynb), I provide a simple sample for comparing the oscillation retrival with Hebbian Learning rule and STDP rule. 

* Customization: You can change the stored complex matrices (files in `src/samples/`), or uncommit the random vector generator to randomly generate testing samples.

# Reference:

[1] GERSTNER W, KISTLER W M, NAUD R, et al. Neuronal dynamics: From single neurons to networks and models of cognition[M]. Cambridge University Press, 2014.

[2] IZHIKEVICH E M. Simple model of spiking neurons[J]. IEEE Transactions on neural networks,
 2003, 14(6):1569-1572.

[3] FANG X, DUAN S,WANGL. Memristive izhikevich spiking neuron model and its application in oscillatory associative memory[J]. Frontiers in Neuroscience, 2022, 16:885322

[4] FRADY E P, SOMMER F T. Robust computation with rhythmic spike patterns[J]. Proceedings of the National Academy of Sciences, 2019, 116(36):18050-18059.