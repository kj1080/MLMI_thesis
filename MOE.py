import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention


# new includes loss to help MOE diversity 
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention

class MoEAttentionProjection(nn.Module):
    def __init__(self, in_features, out_features, num_experts, k=1, 
                 segment_expert_trigger=False, temperature=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.segment_expert_trigger = segment_expert_trigger
        self.temperature = temperature

        self.experts = nn.ModuleList([
            nn.Linear(in_features, out_features) for _ in range(num_experts)
        ])
        for expert in self.experts:
            nn.init.orthogonal_(expert.weight)

        self.gate = nn.Linear(in_features, num_experts)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        # For logging/analysis
        self._last_expert_outputs = None
        self._last_gate_weights = None
        self._last_topk_indices = None
        self._last_topk_scores = None

    def forward(self, x, return_expert_outputs=False):
        # x: [batch, seq_len, hidden]
        temp = getattr(self, "temperature", 1.0)

        if not self.segment_expert_trigger:
            gate_logits = self.gate(x) / temp
            gate_scores = F.softmax(gate_logits, dim=-1)
        else:
            segment_repr = x.mean(dim=1)  # [batch, hidden]
            gate_logits = self.gate(segment_repr) / temp
            gate_scores = F.softmax(gate_logits, dim=-1)
            gate_scores = gate_scores.unsqueeze(1).expand(-1, x.size(1), -1)

        topk_scores, topk_indices = torch.topk(gate_scores, self.k, dim=-1)

        # Save for logging/analysis
        self._last_gate_weights = gate_scores.detach().cpu()
        self._last_topk_indices = topk_indices.detach().cpu()
        self._last_topk_scores = topk_scores.detach().cpu()

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)  # [batch, seq_len, num_experts, hidden]
        self._last_expert_outputs = expert_outputs.detach()

        # Route inputs via top-k experts
        topk_outputs = []
        for i in range(self.k):
            indices = topk_indices[:, :, i].unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1)).unsqueeze(-2)
            selected = torch.gather(expert_outputs, dim=-2, index=indices).squeeze(-2)
            topk_outputs.append(selected * topk_scores[:, :, i].unsqueeze(-1))

        output = sum(topk_outputs)
        if return_expert_outputs:
            return output, expert_outputs
        return output

    def orthogonality_loss(self):
        loss = 0.0
        num_pairs = 0
        for i in range(len(self.experts)):
            for j in range(i + 1, len(self.experts)):
                wi = self.experts[i].weight
                wj = self.experts[j].weight
                wi_flat = wi.view(-1)
                wj_flat = wj.view(-1)
                dot = (wi_flat * wj_flat).sum()
                loss += dot ** 2
                num_pairs += 1
        return loss / num_pairs if num_pairs > 0 else 0.0

    def output_diversity_loss(self, expert_outputs=None):
        if expert_outputs is None:
            expert_outputs = self._last_expert_outputs
        # expert_outputs: [batch, seq_len, num_experts, hidden]
        if expert_outputs.dim() == 4 and expert_outputs.size(0) == 1:
            expert_outputs = expert_outputs.squeeze(0)  # [seq_len, num_experts, hidden]
        loss = 0.0
        num_pairs = 0
        n_experts = expert_outputs.size(1)
        for i in range(n_experts):
            for j in range(i + 1, n_experts):
                out_i = expert_outputs[:, i, :]
                out_j = expert_outputs[:, j, :]
                sim = F.cosine_similarity(out_i, out_j, dim=-1)
                loss += sim.mean()
                num_pairs += 1
        return loss / num_pairs if num_pairs > 0 else 0.0

    def set_temperature(self, temperature):
        self.temperature = temperature

    def get_last_logging_data(self):
        return {
            "expert_outputs": self._last_expert_outputs,
            "gate_weights": self._last_gate_weights,
            "topk_indices": self._last_topk_indices,
            "topk_scores": self._last_topk_scores
        }

class MoEBertSelfAttention(BertSelfAttention):
    def __init__(self, config, num_experts=4, k=1, segment_expert_trigger=False):
        super().__init__(config)
        self.q_proj = MoEAttentionProjection(config.hidden_size, config.hidden_size, num_experts, k, segment_expert_trigger=segment_expert_trigger)
        self.k_proj = MoEAttentionProjection(config.hidden_size, config.hidden_size, num_experts, k, segment_expert_trigger=segment_expert_trigger)
        self.v_proj = MoEAttentionProjection(config.hidden_size, config.hidden_size, num_experts, k, segment_expert_trigger=segment_expert_trigger)

        self.last_gating_weights = None
        self.last_attention_probs = None

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_x_shape).permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, attention_mask=None, head_mask=None,
        encoder_hidden_states=None, encoder_attention_mask=None,
        past_key_value=None, output_attentions=False,
    ):
        mixed_query_layer = self.q_proj(hidden_states)
        self.last_gating_weights = self.q_proj._last_gate_weights.detach().clone()
        self.last_expert_indices = self.q_proj._last_topk_indices

        if encoder_hidden_states is not None:
            mixed_key_layer = self.k_proj(encoder_hidden_states)
            mixed_value_layer = self.v_proj(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.k_proj(hidden_states)
            mixed_value_layer = self.v_proj(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        self.last_attention_probs = attention_probs.detach().clone()
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if self.is_decoder:
            outputs = outputs + (key_layer, value_layer)
        return outputs

class MoERobertaSelfAttention(RobertaSelfAttention):
    def __init__(self, config, num_experts=4, k=1, segment_expert_trigger=False):
        super().__init__(config)
        self.q_proj = MoEAttentionProjection(config.hidden_size, config.hidden_size, num_experts, k, segment_expert_trigger=segment_expert_trigger)
        self.k_proj = MoEAttentionProjection(config.hidden_size, config.hidden_size, num_experts, k, segment_expert_trigger=segment_expert_trigger)
        self.v_proj = MoEAttentionProjection(config.hidden_size, config.hidden_size, num_experts, k, segment_expert_trigger=segment_expert_trigger)

        self.last_gating_weights = None
        self.last_attention_probs = None

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_x_shape).permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, attention_mask=None, head_mask=None,
        encoder_hidden_states=None, encoder_attention_mask=None,
        past_key_value=None, output_attentions=False,
    ):
        mixed_query_layer = self.q_proj(hidden_states)
        self.last_gating_weights = self.q_proj._last_gate_weights.detach().clone()
        self.last_expert_indices = self.q_proj._last_topk_indices

        if encoder_hidden_states is not None:
            mixed_key_layer = self.k_proj(encoder_hidden_states)
            mixed_value_layer = self.v_proj(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.k_proj(hidden_states)
            mixed_value_layer = self.v_proj(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        self.last_attention_probs = attention_probs.detach().clone()
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if self.is_decoder:
            outputs = outputs + (key_layer, value_layer)
        return outputs



# class MoEAttentionProjection(nn.Module):
#     def __init__(self, input_dim, output_dim, num_experts, k=1):
#         super().__init__()
#         self.num_experts = num_experts
#         self.k = k  # top-k experts
#         self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
#         self.gate = nn.Linear(input_dim, num_experts)  # Gating network

#     def forward(self, x):
#         # x: [batch_size, seq_len, input_dim]
#         batch_size, seq_len, dim = x.size()
#         x_flat = x.view(-1, dim)  # [batch_size * seq_len, input_dim]

#         gate_logits = self.gate(x_flat)  # [B*T, num_experts]
#         gate_scores = F.softmax(gate_logits, dim=-1)

#         # Get top-k expert indices and scores
#         topk_scores, topk_indices = torch.topk(gate_scores, self.k, dim=-1)  # [B*T, k]

#         # Create output tensor
#         output = torch.zeros((x_flat.size(0), self.experts[0].out_features), device=x.device)

#         for i in range(self.k):
#             expert_idx = topk_indices[:, i]
#             expert_weight = topk_scores[:, i].unsqueeze(1)

#             # Prepare expert inputs
#             expert_outputs = torch.zeros_like(output)
#             for j in range(self.num_experts):
#                 mask = (expert_idx == j)
#                 if mask.any():
#                     selected = x_flat[mask]
#                     expert_outputs[mask] = self.experts[j](selected)

#             output += expert_outputs * expert_weight

#         return output.view(batch_size, seq_len, -1)  # [B, T, D]


# class MoEBertSelfAttention(BertSelfAttention):
#     def __init__(self, config, num_experts=4, k=1, segment_expert_trigger=True):
#         super().__init__(config)
#         self.q_proj = MoEAttentionProjection(config.hidden_size, config.hidden_size, num_experts, k, segment_expert_trigger=segment_expert_trigger)
#         self.k_proj = MoEAttentionProjection(config.hidden_size, config.hidden_size, num_experts, k, segment_expert_trigger=segment_expert_trigger)
#         self.v_proj = MoEAttentionProjection(config.hidden_size, config.hidden_size, num_experts, k, segment_expert_trigger=segment_expert_trigger)


#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         return x.view(*new_x_shape).permute(0, 2, 1, 3)

#     def forward(
#         self,
#         hidden_states,
#         attention_mask=None,
#         head_mask=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         past_key_value=None,
#         output_attentions=False,
#     ):
#         mixed_query_layer = self.q_proj(hidden_states)
#         if encoder_hidden_states is not None:
#             mixed_key_layer = self.k_proj(encoder_hidden_states)
#             mixed_value_layer = self.v_proj(encoder_hidden_states)
#             attention_mask = encoder_attention_mask
#         else:
#             mixed_key_layer = self.k_proj(hidden_states)
#             mixed_value_layer = self.v_proj(hidden_states)

#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)

#         # Take the dot product between "query" and "key" to get the raw attention scores.
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         if attention_mask is not None:
#             attention_scores = attention_scores + attention_mask

#         # Normalize the attention scores to probabilities.
#         attention_probs = nn.Softmax(dim=-1)(attention_scores)

#         # This is actually dropping out entire tokens to attend to, which might seem a bit unusual,
#         # but is taken from the original Transformer paper.
#         attention_probs = self.dropout(attention_probs)

#         # Mask heads if we want to
#         if head_mask is not None:
#             attention_probs = attention_probs * head_mask

#         context_layer = torch.matmul(attention_probs, value_layer)

#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)

#         outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
#         if self.is_decoder:
#             outputs = outputs + (key_layer, value_layer)

#         return outputs

# class MoERobertaSelfAttention(RobertaSelfAttention):
#     def __init__(self, config, num_experts=4, k=1, segment_expert_trigger=True):
#         super().__init__(config)
#         self.q_proj = MoEAttentionProjection(config.hidden_size, config.hidden_size, num_experts, k, segment_expert_trigger=segment_expert_trigger)
#         self.k_proj = MoEAttentionProjection(config.hidden_size, config.hidden_size, num_experts, k, segment_expert_trigger=segment_expert_trigger)
#         self.v_proj = MoEAttentionProjection(config.hidden_size, config.hidden_size, num_experts, k, segment_expert_trigger=segment_expert_trigger)


#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         return x.view(*new_x_shape).permute(0, 2, 1, 3)

#     def forward(
#         self,
#         hidden_states,
#         attention_mask=None,
#         head_mask=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         past_key_value=None,
#         output_attentions=False,
#     ):
#         mixed_query_layer = self.q_proj(hidden_states)
#         if encoder_hidden_states is not None:
#             mixed_key_layer = self.k_proj(encoder_hidden_states)
#             mixed_value_layer = self.v_proj(encoder_hidden_states)
#             attention_mask = encoder_attention_mask
#         else:
#             mixed_key_layer = self.k_proj(hidden_states)
#             mixed_value_layer = self.v_proj(hidden_states)

#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)

#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         if attention_mask is not None:
#             attention_scores = attention_scores + attention_mask

#         attention_probs = nn.Softmax(dim=-1)(attention_scores)
#         attention_probs = self.dropout(attention_probs)

#         if head_mask is not None:
#             attention_probs = attention_probs * head_mask

#         context_layer = torch.matmul(attention_probs, value_layer)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)

#         outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
#         if self.is_decoder:
#             outputs = outputs + (key_layer, value_layer)

#         return outputs

# class LoRALinear(nn.Module):
#     def __init__(self, in_features, out_features, r=4, alpha=16, dropout=0.1):
#         super().__init__()
#         self.r = r
#         self.lora_A = nn.Linear(in_features, r, bias=False)
#         self.lora_B = nn.Linear(r, out_features, bias=False)
#         self.scale = alpha / r
#         self.dropout = nn.Dropout(dropout)

#         # Optionally freeze the base
#         for p in self.lora_A.parameters():
#             nn.init.kaiming_uniform_(p, a=math.sqrt(5))
#         for p in self.lora_B.parameters():
#             nn.init.zeros_(p)

#     def forward(self, x):
#         return self.dropout(self.lora_B(self.lora_A(x))) * self.scale




#### new with ability to trigger segment level rather than token level #######
# import torch
# import math
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers.models.bert.modeling_bert import BertSelfAttention
# from transformers.models.roberta.modeling_roberta import RobertaSelfAttention

# # ==== UPDATED: Add segment_expert_trigger to MoEAttentionProjection ====
# class MoEAttentionProjection(nn.Module):
#     def __init__(self, input_dim, output_dim, num_experts, k=1, segment_expert_trigger=True):
#         super().__init__()
#         self.num_experts = num_experts
#         self.k = k  # top-k experts
#         self.segment_expert_trigger = segment_expert_trigger
#         self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
#         self.gate = nn.Linear(input_dim, num_experts)  # Gating network

#     def forward(self, x):
#         # x: [batch_size, seq_len, input_dim]
#         batch_size, seq_len, dim = x.size()
#         if not self.segment_expert_trigger:
#             # Token-level gating (same as before)
#             x_flat = x.view(-1, dim)  # [batch_size * seq_len, input_dim]
#             gate_logits = self.gate(x_flat)  # [B*T, num_experts]
#             gate_scores = F.softmax(gate_logits, dim=-1)
#             topk_scores, topk_indices = torch.topk(gate_scores, self.k, dim=-1)  # [B*T, k]
#             output = torch.zeros((x_flat.size(0), self.experts[0].out_features), device=x.device)
#             for i in range(self.k):
#                 expert_idx = topk_indices[:, i]
#                 expert_weight = topk_scores[:, i].unsqueeze(1)
#                 expert_outputs = torch.zeros_like(output)
#                 for j in range(self.num_experts):
#                     mask = (expert_idx == j)
#                     if mask.any():
#                         selected = x_flat[mask]
#                         expert_outputs[mask] = self.experts[j](selected)
#                 output += expert_outputs * expert_weight
#             return output.view(batch_size, seq_len, -1)
#         else:
#             # ==== Segment-level gating ====
#             segment_repr = x.mean(dim=1)  # [batch, input_dim]
#             gate_logits = self.gate(segment_repr)  # [batch, num_experts]
#             gate_scores = F.softmax(gate_logits, dim=-1)  # [batch, num_experts]
#             # For each batch, select top-k experts
#             topk_scores, topk_indices = torch.topk(gate_scores, self.k, dim=-1)  # [batch, k]
#             output = torch.zeros((batch_size, seq_len, self.experts[0].out_features), device=x.device)
#             # For each example in the batch
#             for b in range(batch_size):
#                 for i in range(self.k):
#                     expert_j = topk_indices[b, i].item()
#                     weight = topk_scores[b, i]
#                     output[b] += self.experts[expert_j](x[b]) * weight  # [seq_len, out_dim]
#             return output  # [batch, seq_len, out_dim]


# class MoEBertSelfAttention(BertSelfAttention):
#     def __init__(self, config, num_experts=4, k=1, segment_expert_trigger=False):
#         super().__init__(config)
#         self.q_proj = MoEAttentionProjection(config.hidden_size, config.hidden_size, num_experts, k, segment_expert_trigger=segment_expert_trigger)
#         self.k_proj = MoEAttentionProjection(config.hidden_size, config.hidden_size, num_experts, k, segment_expert_trigger=segment_expert_trigger)
#         self.v_proj = MoEAttentionProjection(config.hidden_size, config.hidden_size, num_experts, k, segment_expert_trigger=segment_expert_trigger)

#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         return x.view(*new_x_shape).permute(0, 2, 1, 3)

#     def forward(
#         self,
#         hidden_states,
#         attention_mask=None,
#         head_mask=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         past_key_value=None,
#         output_attentions=False,
#     ):
#         mixed_query_layer = self.q_proj(hidden_states)
#         if encoder_hidden_states is not None:
#             mixed_key_layer = self.k_proj(encoder_hidden_states)
#             mixed_value_layer = self.v_proj(encoder_hidden_states)
#             attention_mask = encoder_attention_mask
#         else:
#             mixed_key_layer = self.k_proj(hidden_states)
#             mixed_value_layer = self.v_proj(hidden_states)
#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         if attention_mask is not None:
#             attention_scores = attention_scores + attention_mask
#         attention_probs = nn.Softmax(dim=-1)(attention_scores)
#         attention_probs = self.dropout(attention_probs)
#         if head_mask is not None:
#             attention_probs = attention_probs * head_mask
#         context_layer = torch.matmul(attention_probs, value_layer)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)
#         outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
#         if self.is_decoder:
#             outputs = outputs + (key_layer, value_layer)
#         return outputs


# class MoERobertaSelfAttention(RobertaSelfAttention):
#     def __init__(self, config, num_experts=4, k=1, segment_expert_trigger=True):
#         super().__init__(config)
#         self.q_proj = MoEAttentionProjection(config.hidden_size, config.hidden_size, num_experts, k, segment_expert_trigger=segment_expert_trigger)
#         self.k_proj = MoEAttentionProjection(config.hidden_size, config.hidden_size, num_experts, k, segment_expert_trigger=segment_expert_trigger)
#         self.v_proj = MoEAttentionProjection(config.hidden_size, config.hidden_size, num_experts, k, segment_expert_trigger=segment_expert_trigger)

#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         return x.view(*new_x_shape).permute(0, 2, 1, 3)

#     def forward(
#         self,
#         hidden_states,
#         attention_mask=None,
#         head_mask=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         past_key_value=None,
#         output_attentions=False,
#     ):
#         mixed_query_layer = self.q_proj(hidden_states)
#         if encoder_hidden_states is not None:
#             mixed_key_layer = self.k_proj(encoder_hidden_states)
#             mixed_value_layer = self.v_proj(encoder_hidden_states)
#             attention_mask = encoder_attention_mask
#         else:
#             mixed_key_layer = self.k_proj(hidden_states)
#             mixed_value_layer = self.v_proj(hidden_states)
#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         if attention_mask is not None:
#             attention_scores = attention_scores + attention_mask
#         attention_probs = nn.Softmax(dim=-1)(attention_scores)
#         attention_probs = self.dropout(attention_probs)
#         if head_mask is not None:
#             attention_probs = attention_probs * head_mask
#         context_layer = torch.matmul(attention_probs, value_layer)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)
#         outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
#         if self.is_decoder:
#             outputs = outputs + (key_layer, value_layer)
#         return outputs
