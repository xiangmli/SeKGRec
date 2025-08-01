# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from activations import ACT2FN
from moe_config import MoEConfig
import torch.nn.functional as F
import pdb


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates, router_type):
        """Create a SparseDispatcher."""
        # each forward pass initialize the sparse dispatcher once
        self._gates = gates
        self._num_experts = num_experts
        self._router_type = router_type
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        # _batch_index: sample index inside a batch that is assigned to particular expert, concatenated
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        # from e.g., [64, 4] -> [256, 4], collection of samples assigned to each gate
        gates_exp = gates[self._batch_index.flatten()]
        # difference between torch.nonzero(gates) and self._nonzero_gates
        # the first one is index and the second one is actural weights!
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        if isinstance(inp, list) and self._router_type == 'joint':
            inp = torch.concat(inp, dim=1)
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        # concat all sample outputs from each expert
        stitched = torch.cat(expert_out, 0).exp()
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        # this is the weighted combination step
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MLP(nn.Module):
    def __init__(self, config: MoEConfig, input_size: int, output_size: int, hidden_size):
        super(MLP, self).__init__()

        if isinstance(hidden_size, int):
            hidden_sizes = [hidden_size]
        else:
            hidden_sizes = hidden_size

        layer_sizes = [input_size] + hidden_sizes + [output_size]

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.dropout = nn.Dropout(config.dropout)
        self.activation = ACT2FN[config.hidden_act]
        self.log_soft = nn.LogSoftmax(1)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)

        x = self.layers[-1](x)
        x = self.log_soft(x)
        return x



class MoE(nn.Module):
    def __init__(self, config: MoEConfig):
        super(MoE, self).__init__()
        self.noisy_gating = config.noisy_gating
        self.num_experts = config.num_experts
        self.output_size = config.moe_output_size
        self.input_size = config.moe_input_size
        self.hidden_size = config.moe_hidden_size
        self.k = config.top_k
        self.disjoint_k = config.disjoint_top_k
        self.router_type = config.router_type
        self.num_modalities = config.num_modalities
        self.gating = config.gating
        self.loss_coef = config.loss_coef

        if self.router_type == 'disjoint':
            self.w_gate = [nn.Parameter(
                torch.zeros(self.input_size // self.num_modalities, self.num_experts // self.num_modalities),
                requires_grad=True) for _ in range(self.num_modalities)]
            self.w_noise = [nn.Parameter(
                torch.zeros(self.input_size // self.num_modalities, self.num_experts // self.num_modalities),
                requires_grad=True) for _ in range(self.num_modalities)]
        elif self.router_type == 'permod':
            self.w_gate = [
                nn.Parameter(torch.zeros(self.input_size // self.num_modalities, self.num_experts), requires_grad=True)
                for _ in range(self.num_modalities)]
            self.w_noise = [
                nn.Parameter(torch.zeros(self.input_size // self.num_modalities, self.num_experts), requires_grad=True)
                for _ in range(self.num_modalities)]
        else:
            self.w_gate = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)
            self.w_noise = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)

        if self.router_type == 'disjoint':
            self.experts = nn.ModuleList(
                nn.ModuleList(
                    [MLP(config, self.input_size // self.num_modalities, self.output_size, self.hidden_size)
                     for _ in range(self.num_experts // self.num_modalities)])
                for _ in range(self.num_modalities)
            )
        elif self.router_type == 'permod':
            self.experts = nn.ModuleList(
                [MLP(config, self.input_size // self.num_modalities, self.output_size, self.hidden_size)
                 for _ in range(self.num_experts)])
        else:
            self.experts = nn.ModuleList(
                [MLP(config, self.input_size, self.output_size, self.hidden_size)
                 for _ in range(self.num_experts)])

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.Tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        if self.router_type == 'disjoint':
            threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.disjoint_k
        else:
            threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k

        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def _get_logits(self, x, train, noise_epsilon, idx=None):
        if idx is not None:
            w_gate = self.w_gate[idx].to(x.device)
            w_noise = self.w_noise[idx].to(x.device)
        else:
            w_gate = self.w_gate
            w_noise = self.w_noise

        if self.gating == 'softmax':
            clean_logits = x @ w_gate
        elif self.gating == 'laplace':
            clean_logits = -torch.cdist(x, torch.t(w_gate))
        elif self.gating == 'gaussian':
            clean_logits = -torch.pow(torch.cdist(x, torch.t(w_gate)), 2)

        if self.noisy_gating:
            raw_noise_stddev = x @ w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train)
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
            noisy_logits = None
            noise_stddev = None
        return logits, clean_logits, noisy_logits, noise_stddev

    def _top_k_gating(self, logits, clean_logits, noisy_logits, noise_stddev, k):
        top_logits, top_indices = logits.topk(min(k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :k]
        top_k_indices = top_indices[:, :k]

        if self.gating == 'softmax':
            top_k_gates = self.softmax(top_k_logits)
        elif self.gating == 'laplace' or self.gating == 'gaussian':
            top_k_gates = torch.exp(top_k_logits - torch.logsumexp(top_k_logits, dim=1, keepdim=True))

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and k < self.num_experts and self.training:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2, modalities=None):
        if self.router_type == 'joint':
            if isinstance(x, list):
                embeddings = torch.concat(x, dim=1)
            else:
                embeddings = x
            logits, clean_logits, noisy_logits, noise_stddev = self._get_logits(embeddings, train, noise_epsilon)
            gates, load = self._top_k_gating(logits, clean_logits, noisy_logits, noise_stddev, self.k)
            return gates, load
        else:
            all_gates, all_loads = [], []
            for i in range(self.num_modalities):
                logits, clean_logits, noisy_logits, noise_stddev = self._get_logits(x[i], train, noise_epsilon, idx=i)
                if self.router_type == 'permod':
                    gates, load = self._top_k_gating(logits, clean_logits, noisy_logits, noise_stddev, self.k)
                else:
                    gates, load = self._top_k_gating(logits, clean_logits, noisy_logits, noise_stddev, self.disjoint_k)
                all_gates.append(gates)
                all_loads.append(load)
            return all_gates, all_loads

    def forward(self, x, train=True, loss_coef=1e-2, modalities=None):
        gates, load = self.noisy_top_k_gating(x, train, modalities=modalities)
        loss_coef = self.loss_coef
        if isinstance(gates, list):
            loss, y = 0, 0
            sub_experts = self.num_experts // self.num_modalities
            for g, l in zip(gates, load):
                loss += self.cv_squared(g.sum(0)) + self.cv_squared(l)
            loss *= loss_coef
            for j, g in enumerate(gates):
                dispatcher = SparseDispatcher(self.num_experts, g, self.router_type)
                expert_inputs = dispatcher.dispatch(x[j])
                if self.router_type == 'permod':
                    expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
                elif self.router_type == 'disjoint':
                    expert_outputs = [self.experts[j][i](expert_inputs[i]) for i in range(sub_experts)]
                y += dispatcher.combine(expert_outputs)
        else:
            loss = self.cv_squared(gates.sum(0)) + self.cv_squared(load)
            loss *= loss_coef
            dispatcher = SparseDispatcher(self.num_experts, gates, self.router_type)
            expert_inputs = dispatcher.dispatch(x)
            expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
            y = dispatcher.combine(expert_outputs)
        return y, loss



class RecMoE(nn.Module):
    def __init__(self, user_moe_config, item_moe_config, embedding_dim=64):
        super(RecMoE, self).__init__()

        self.user_moe = MoE(user_moe_config)
        self.item_moe = MoE(item_moe_config)

        self.embedding_dim = embedding_dim

        assert user_moe_config.moe_output_size == item_moe_config.moe_output_size, \
            "User MoE and Item MoE must have the same output dimension"

        self.num_neg_sample = 5
        self.margin_ccl = 0.5

    def forward(self, user_features, item_features, train=True, loss_coef=1e-2):
        user_output, user_moe_loss = self.user_moe(user_features, train=train, loss_coef=loss_coef)
        item_output, item_moe_loss = self.item_moe(item_features, train=train, loss_coef=loss_coef)

        user_embeddings = user_output
        item_embeddings = item_output

        user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
        item_embeddings = F.normalize(item_embeddings, p=2, dim=1)

        return user_embeddings, item_embeddings, user_moe_loss, item_moe_loss

    def predict_score(self, user_features, item_features, train=False):
        user_embeddings, item_embeddings, _, _ = self.forward(
            user_features, item_features, train=train
        )
        scores = torch.sum(user_embeddings * item_embeddings, dim=1)
        return scores

    def bpr_loss(self, users, pos_items, neg_items, user_features, item_features,
                 moe_loss_coef=1e-2, bpr_loss_coef=1.0):
        user_features_indexed = [feat[users] for feat in user_features]
        users_emb, user_moe_loss = self.user_moe(user_features_indexed, train=True, loss_coef=moe_loss_coef)

        pos_item_features_indexed = [feat[pos_items] for feat in item_features]
        pos_emb, pos_item_moe_loss = self.item_moe(pos_item_features_indexed, train=True, loss_coef=moe_loss_coef)

        neg_item_features_indexed = [feat[neg_items] for feat in item_features]
        neg_emb, neg_item_moe_loss = self.item_moe(neg_item_features_indexed, train=True, loss_coef=moe_loss_coef)

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        moe_loss = user_moe_loss + pos_item_moe_loss + neg_item_moe_loss
        total_loss = bpr_loss_coef * bpr_loss + moe_loss_coef * moe_loss

        return total_loss

    def create_contrastive_loss(self, u_e, pos_e, neg_e):
        batch_size = u_e.shape[0]

        u_e = F.normalize(u_e)
        pos_e = F.normalize(pos_e)
        neg_e = F.normalize(neg_e)

        ui_pos_loss1 = torch.relu(1 - torch.cosine_similarity(u_e, pos_e, dim=1))

        users_batch = torch.repeat_interleave(u_e, self.num_neg_sample, dim=0)

        ui_neg1 = torch.relu(torch.cosine_similarity(users_batch, neg_e, dim=1) - self.margin_ccl)
        ui_neg1 = ui_neg1.view(batch_size, -1)
        x = ui_neg1 > 0
        ui_neg_loss1 = torch.sum(ui_neg1, dim=-1) / (torch.sum(x, dim=-1) + 1e-5)

        loss = ui_pos_loss1 + ui_neg_loss1

        return loss.mean()

    def contrastive_loss(self, users, pos_items, neg_items, user_features, item_features,
                         moe_loss_coef=1e-2, contrastive_loss_coef=1.0):
        user_features_indexed = [feat[users] for feat in user_features]
        users_emb, user_moe_loss = self.user_moe(user_features_indexed, train=True, loss_coef=moe_loss_coef)

        pos_item_features_indexed = [feat[pos_items] for feat in item_features]
        pos_emb, pos_item_moe_loss = self.item_moe(pos_item_features_indexed, train=True, loss_coef=moe_loss_coef)

        batch_size = users.shape[0]
        expected_neg_size = batch_size * self.num_neg_sample
        if neg_items.shape[0] != expected_neg_size:
            if neg_items.shape[0] > expected_neg_size:
                neg_items = neg_items[:expected_neg_size]
            else:
                repeat_times = expected_neg_size // neg_items.shape[0] + 1
                neg_items = neg_items.repeat(repeat_times)[:expected_neg_size]

        neg_item_features_indexed = [feat[neg_items] for feat in item_features]
        neg_emb, neg_item_moe_loss = self.item_moe(neg_item_features_indexed, train=True, loss_coef=moe_loss_coef)

        contrastive_loss = self.create_contrastive_loss(users_emb, pos_emb, neg_emb)
        moe_loss = user_moe_loss + pos_item_moe_loss + neg_item_moe_loss
        total_loss = contrastive_loss_coef * contrastive_loss + moe_loss_coef * moe_loss

        return total_loss




