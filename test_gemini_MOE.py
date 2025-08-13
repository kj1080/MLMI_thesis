import torch

from gemini_lora_MOE import load_test_set()

print(load_test_set())

# def test_topkmoe_forward_output_shape():
#     d_model = 768
#     n_heads = 12
#     num_experts = 4
#     moe = TopKMoEAttention(d_model, n_heads, num_experts)

#     B, T = 2, 128
#     dummy_input = torch.randn(B, T, d_model)
#     attention_mask = torch.ones(B, T)

#     output, _ = moe(dummy_input, attention_mask)
    
#     assert output.last_hidden_state.shape == (B, T, d_model), "Output shape mismatch"

#     assert not torch.isnan(output.last_hidden_state).any(), "NaN values in output"


# def test_gating_weights_sum_to_one():
#     d_model = 768
#     n_heads = 12
#     num_experts = 3
#     moe = TopKMoEAttention(d_model, n_heads, num_experts, top_k=2)

#     B, T = 4, 64
#     dummy_input = torch.randn(B, T, d_model)
#     mask = torch.ones(B, T)

#     moe.eval()
#     moe(dummy_input, attention_mask=mask)

#     assert moe.last_gating is not None, "Gating not computed"
#     gating_sums = moe.last_gating.sum(dim=-1)
#     assert torch.allclose(gating_sums, torch.ones_like(gating_sums)), "Gating weights do not sum to 1"

# # def test_empty_input_handling():
# #     d_model = 768
# #     n_heads = 12
# #     num_experts = 2
# #     moe = TopKMoEAttention(d_model, n_heads, num_experts)

# #     B, T = 2, 128
# #     dummy_input = torch.zeros(B, T, d_model)
# #     mask = torch.zeros(B, T)

# #     output, aux_loss = moe(dummy_input, attention_mask=mask)

# #     # Now `output` is the tensor you care about directly
# #     assert output.shape == (B, T, d_model), "Shape mismatch on empty input"
# #     assert aux_loss is not None and isinstance(aux_loss, torch.Tensor), "Aux loss should be a defined torch.Tensor"



# def test_topkmoeattention_gradient_flow():
#     d_model = 768
#     n_heads = 12
#     num_experts = 2
#     B, T = 2, 16

#     moe = TopKMoEAttention(d_model, n_heads, num_experts)
#     moe.train()  # Set training mode for dropout and wandb logging

#     dummy_input = torch.randn(B, T, d_model, requires_grad=True)
#     mask = torch.ones(B, T)

#     output, aux_loss = moe(dummy_input, attention_mask=mask)
#     last_hidden = output.last_hidden_state  # Correct way to extract the tensor

#     dummy_loss = last_hidden.mean() + 0.1 * aux_loss

#     dummy_loss.backward()

#     grads_found = False
#     for name, param in moe.named_parameters():
#         if param.requires_grad:
#             assert param.grad is not None, f"Parameter {name} has no gradient"
#             assert param.grad.abs().sum() > 0, f"Parameter {name} has zero gradient"
#             grads_found = True

#     assert grads_found, "No trainable parameters found with gradients"


