import torch
import re

def lambda_return(reward, value, discount, bootstrap, lambda_):
    """
    Computes the lambda-return, a key component for training the critic.
    """
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + discount * next_values * (1 - lambda_)
    
    last = bootstrap
    returns = []
    for i in reversed(range(len(inputs))):
        last = inputs[i] + discount[i] * lambda_ * last
        returns.append(last)
    
    return torch.stack(list(reversed(returns)), dim=0)

def schedule(string, step):
    try:
        return float(string)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, duration = [float(g) for g in match.groups()]
            mix = torch.clamp(torch.tensor(step / duration), 0, 1)
            return (1 - mix) * initial + mix * final
        raise NotImplementedError(string)
