"""Token-level advantage estimation for a completed response."""
import torch
import torch.nn.functional as F

def gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate advantages and unnormalized returns with zero terminal value."""
    values = values.detach().float()
    rewards = rewards.detach().float()
    advantages = torch.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(rewards.numel())):
        next_value = values[t + 1] if t + 1 < rewards.numel() else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * lam * last
        advantages[t] = last
    return advantages, advantages + values


def rollout_logps_and_values(policy, value_head, seq, prompt_len, grad):
    with torch.enable_grad() if grad else torch.no_grad():
        out=policy(input_ids=seq,attention_mask=torch.ones_like(seq),output_hidden_states=True,return_dict=True,use_cache=False)
        # Prediction at position t-1 assigns probability to token t.
        logp=F.log_softmax(out.logits[:,:-1,:].float(),dim=-1).gather(-1,seq[:,1:].unsqueeze(-1)).squeeze(-1)
        val=value_head(out.hidden_states[-1][:,:-1,:].float()).squeeze(-1)
        return logp[:,prompt_len-1:].squeeze(0),val[:,prompt_len-1:].squeeze(0)

def only_logps(ref,seq,prompt_len):
    with torch.no_grad():
        out=ref(input_ids=seq,attention_mask=torch.ones_like(seq),return_dict=True,use_cache=False)
        logp=F.log_softmax(out.logits[:,:-1,:].float(),dim=-1).gather(-1,seq[:,1:].unsqueeze(-1)).squeeze(-1)
        return logp[:,prompt_len-1:].squeeze(0)

