# coding = utf-8
# Copyright 2025 Rikka Botan. All rights reserved
# Licensed under "MIT License"
# IMoE: Indefinacy Mixture of Experts

import torch
from torch import nn
import torch.nn.functional as F
from typing import Any
import math

"""
IMoE: Indefinacy Mixture of Experts architecture dynamically selects neurons.
This implementation replicates the neurons
which is dynamically changing synapses
and enables efficient and diverse inference.

Appendix A: IMoE theory

Realization of the neural network based on random graph theory.
The growth process of brain neural circuits can be explained by the theory
that integrates Erdős-Rényi-Gilbert model
and fitness model (Bianconi-Barabási model).

During childhood, the brain neural circuits and synapses increase rapidly
and then synapses are pruned as the brain grows.
This growth process is equivalent to applying Erdős-Rényi-Gilbert model
then fitness model (Bianconi-Barabási model).

Step 1: Erdős-Rényi-Gilbert model like growth process

Individual synapses are not affected by the state of other synapses
and are probabilistically formed.
This phenomenon is represented by Erdős-Rényi-Gilbert model
and is achieved in the algorithm by a parallel definition of the modules.

Step 2: fitness model (Bianconi-Barabási model) like routing

Individual neurons have link coefficients
which affect connected synapses architecture.
These link coefficients change dynamically in response to the environment.
These link coefficients are a random distribution in childhood,
but converge to a constant distribution as they grow older.
This mechanism is realized by a dynamic, multi-level branching process
using softmax functions and non linear projections.

> gate = Softmax(Linear(x))

> x = Top-k(gate) * x
"""

def get_top_p(
    x: torch.Tensor,
    top_p: float = 0.3,
    temperature: float = 1.0,
    dim: int = -1,
    noise: float = 0.1,
    training: bool = False
) -> torch.Tensor:
    """
    ## The function of getting Top-p

    get vals and indices according to temperature

    outputs:
        top_p_vals: coefficients of each experts (experts score)
        top_p_indices: indices of experts
    """
    bsz, seql, _ = x.size()
    x = F.softmax(x, dim=dim).reshape(bsz*seql, -1)
    if training:
        if noise != 0:
            x = x + noise * top_p * torch.randn_like(x)
    if temperature != 1.0:
        x = x / temperature

    if top_p >= 1.0:
        ValueError('top_p should be less than 1.0. default value is 0.3.')

    top_p_indices = x > top_p

    return x, top_p_indices


class IMoE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        inter_dim: int,
        gate_num: int,
        top_p: float = 0.3,
        temperature: float = 1.0,
        noise: float = 0.1,
        bias: bool = False,
        device: Any | None = None,
        dtype: Any | None = None
    ):
        """
        ## IMoE: Indefinacy Mixture of Experts

        Fast Sparse Mixture of Experts

        IMoE module changes its own architecture depending on confidence

        IMoE computes confidence from imformation variance
        """
        super().__init__()
        self.inter_dim = inter_dim
        self.gate_num = gate_num
        self.top_p = top_p
        self.temperature = temperature
        self.noise = noise
        self.experts = nn.ModuleList([
            nn.Linear(
                in_features=input_dim,
                out_features=inter_dim,
                bias=bias,
                device=device,
                dtype=dtype
            )
            for _ in range(gate_num)
        ])
        self.gate = nn.Linear(
            in_features=input_dim,
            out_features=gate_num,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.out_linear = nn.Linear(
            in_features=inter_dim,
            out_features=input_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )
        nn.init.kaiming_normal_(self.gate.weight, a=math.sqrt(0.5))
        nn.init.kaiming_normal_(self.out_linear.weight, a=math.sqrt(0.5))
        for exp in self.experts:
            nn.init.kaiming_normal_(exp.weight, a=math.sqrt(0.5))


    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        bsz, seql, embs = x.size()
        gate_scores = self.gate(x)

        weight_probs, top_p_indices = get_top_p(
            x = gate_scores,
            top_p=self.top_p,
            temperature=self.temperature,
            dim=-1,
            noise=self.noise,
            training=self.training
        )

        weight_probs = weight_probs.to(x.dtype)

        final_hidden_states = torch.zeros(
            (bsz*seql, self.inter_dim), dtype=x.dtype, device=x.device
        )

        for expert_idx in range(self.gate_num):
            mask = (top_p_indices == expert_idx).nonzero(as_tuple=True)[0]
            if mask.numel() == 0:
                continue
            current_state = x.reshape(-1, embs).index_select(0, mask)

            current_hidden_states = (
                self.experts[expert_idx](current_state)
                * weight_probs[:, expert_idx].index_select(0, mask).unsqueeze(-1))

            final_hidden_states.index_add_(
                0,
                mask,
                current_hidden_states.to(x.dtype))

        outputs = final_hidden_states.reshape(bsz, seql, -1)
        outputs = self.out_linear(outputs)

        return outputs
