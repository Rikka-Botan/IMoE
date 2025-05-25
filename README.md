# IMoE: Indefinacy Mixture of Experts
Pytorch official implementation

## About

IMoE dynamically selects neurons.

This implementation replicates the neurons which is dynamically changing synapses and enables efficient and diverse inference.


## Key features

1. Structure that dynamically changes shape depending on input

2. The number of reference memory varies depending on the confidence level of the model

3. Minimizing the number of accesses to memory for faster processing on the GPU

4. Architecture based on random graph theory

## Appendix A: IMoE theory

Realization of the neural network based on random graph theory.
The growth process of brain neural circuits can be explained by the theory
that integrates Erdős-Rényi-Gilbert model
and fitness model (Bianconi-Barabási model).

During childhood, the brain neural circuits and synapses increase rapidly
and then synapses are pruned as the brain grows.
This growth process is equivalent to applying Erdős-Rényi-Gilbert model
then fitness model (Bianconi-Barabási model).

### Step 1: Erdős-Rényi-Gilbert model like growth process

Individual synapses are not affected by the state of other synapses
and are probabilistically formed.
This phenomenon is represented by Erdős-Rényi-Gilbert model
and is achieved in the algorithm by a parallel definition of the modules.

### Step 2: fitness model (Bianconi-Barabási model) like routing

Individual neurons have link coefficients
which affect connected synapses architecture.
These link coefficients change dynamically in response to the environment.
These link coefficients are a random distribution in childhood,
but converge to a constant distribution as they grow older.
This mechanism is realized by a dynamic, multi-level branching process
using softmax functions and linear projections.

***
### Formulation

```math
\displaylines{
W_g \in \mathbb{R^{M \times N}}, \; W_o \in \mathbb{R^{N \times L}}, \; x \in \mathbb{R^N} \\
E = \left( f_1, f_2, \cdots f_G \right) \\
\forall f_i (f_i \in \mathbb{R^{L \times N}}) \\
s: \; gate \; thres \\
Softmax(x) = \frac{e^{x_k}}{\sum_{k=0}^K e^{x_k}} \\
G(x) = Softmax(W_g x) \\
\xi_k(x), \; \Gamma_k(x) = TopP(G(x), s) \\
y = W_o \sum_{i \in \Gamma_k(x)}\xi_i(x)f_i(x)
}
```


## Implementation and License

This repository is official pure pytorch implementation.

Licensed under ["MIT License"](https://mit-license.org/).

Commercial use permitted

## How to use

- Clone the repository

```bash
git clone https://github.com/Rikka-Botan/IMoE.git
```

- Model create

```python
"""
Args:
input_dim: int
inter_dim: int
gate_num: int
top_p: float (default: 0.3)
temperature: float (default: 1.0)
noise: float (default: 0.1) - gating noise parameter when training
bias: bool (default: False)
device: Any | None (default: None)
dtype: Any | None (default: None)
"""

from model.IMoE_modeling import BotanIMoE

model = BotanIMoE(
  768, 768, 16
)
output = model(hidden_states)
```

## Acknowledgements

I thank the developers of python and pytorch.

I thank all the researchers for their efforts to date.

I thank Japan's high standard of education.

And most of all, thank you for your interest in this repository.

## Citations

I would be happy to include a citation at the end, but it is not required.

Feel free to use this model.


## Contact Us

[My X account](https://x.com/peony__snow)


## About Author

### Rikka Botan

Japanese independent researcher having shy and pampered personality >_<

Twin-tail hair is a charm point :)

Interested in natural language processings. 

Usually using python and C.

![RikkaBotan_Logo](https://github.com/user-attachments/assets/92913f91-9136-4d44-8b4d-8a2120118a05)
