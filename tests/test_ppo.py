import torch
from src.ppo import gae


def test_terminal_return_and_backward_credit():
    rewards = torch.tensor([0., 1.])
    values = torch.tensor([.25, .5], requires_grad=True)
    advantages, returns = gae(rewards, values, gamma=1., lam=1.)
    torch.testing.assert_close(advantages, torch.tensor([.75, .5]))
    torch.testing.assert_close(returns, torch.ones(2))
    assert not advantages.requires_grad



def test_checkpointed_policy_receives_gradient():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from peft import get_peft_model
    from src.common import lora
    from src.ppo import rollout_logps_and_values

    torch.manual_seed(42)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_attention_heads=2, num_key_value_heads=1, num_hidden_layers=1)
    policy = get_peft_model(Qwen2ForCausalLM(config), lora())
    policy.enable_input_require_grads()
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for module in policy.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    head = torch.nn.Linear(config.hidden_size, 1)
    tokens = torch.tensor([[1, 3, 5, 7]])
    old, _ = rollout_logps_and_values(policy, head, tokens, 2, grad=False)
    new, _ = rollout_logps_and_values(policy, head, tokens, 2, grad=True)
    assert new.shape == (2,)
    torch.testing.assert_close(old, new)
    ratio = (new - old).exp()
    loss = -(ratio * torch.tensor([1., -1.])).mean()
    loss.backward()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1.5e-5)
    optimizer.step()
    assert any(p.count_nonzero() for name, p in policy.named_parameters() if "lora_B" in name)
