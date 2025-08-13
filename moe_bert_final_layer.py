import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizer

class MoELinear(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        gate_logits = self.gate(x)  # [batch, seq_len, num_experts]
        gate_weights = F.softmax(gate_logits, dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        output = torch.einsum('bsoe,bse->bso', expert_outputs, gate_weights)
        return output, gate_weights  # return gate weights for monitoring

class MoEAttentionLayer(nn.Module):
    def __init__(self, config, num_experts=4):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.query = MoELinear(self.hidden_size, self.hidden_size, num_experts)
        self.key = MoELinear(self.hidden_size, self.hidden_size, num_experts)
        self.value = MoELinear(self.hidden_size, self.hidden_size, num_experts)

        self.out = nn.Linear(self.hidden_size, self.hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        q, q_gate = self.query(hidden_states)
        k, k_gate = self.key(hidden_states)
        v, v_gate = self.value(hidden_states)

        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_shape)
        output = self.out(context_layer)

        return output, {'query': q_gate, 'key': k_gate, 'value': v_gate}

class BertWithMoE(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_experts=4):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        for param in self.bert.parameters():  # freeze all BERT layers
            param.requires_grad = False
        self.moe_layer = MoEAttentionLayer(self.bert.config, num_experts=num_experts)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # binary classification

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        moe_output, gates = self.moe_layer(last_hidden, attention_mask.unsqueeze(1).unsqueeze(2))
        cls_output = moe_output[:, 0, :]  # [CLS] token output
        logits = self.classifier(cls_output)
        return logits, gates
