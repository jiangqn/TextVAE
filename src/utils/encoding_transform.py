import torch

def interpolate(encoding: torch.Tensor, direction: torch.Tensor, scope: float, intervals: int) -> torch.Tensor:
    '''
    :param encoding: torch.FloatTensor (num_layers, num, hidden_size)
    :param direction: torch.FloatTensor (encoding_size,) where encoding_size = num_layers * hidden_size
    :param scope: float
    :param intervals: int
    :return encoding: torch.FloatTensor (num_layers, num * intervals, hidden_size)
    '''

    num_layers, sample_num, hidden_size = encoding.size()
    encoding = encoding.transpose(0, 1).reshape(sample_num, -1)
    direction = direction.unsqueeze(-1)
    projection = encoding.matmul(direction)
    start_encoding = encoding + (-scope - projection) * direction.transpose(0, 1)
    end_encoding = encoding + (scope - projection) * direction.transpose(0, 1)
    weights = torch.arange(0, 1, 1 / intervals, device=encoding.device)
    start_encoding = start_encoding.unsqueeze(1).repeat(1, intervals, 1)
    end_encoding = end_encoding.unsqueeze(1).repeat(1, intervals, 1)
    weights = weights.unsqueeze(0).unsqueeze(-1)
    encoding = start_encoding * (1 - weights) + end_encoding * weights
    encoding = encoding.reshape(sample_num * intervals, -1)
    encoding = encoding.reshape(sample_num * intervals, num_layers, hidden_size).transpose(0, 1)
    return encoding

def move(encoding: torch.Tensor, target_projection: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    '''
    :param encoding: torch.FloatTensor (num, encoding_size)
    :param target_projection: torch.FloatTensor (num,)
    :param direction: torch.FloatTensor (encoding_size,)
    :return encoding: torch.FloatTensor (num, encoding_size)
    '''

    direction = direction.unsqueeze(-1)
    current_projection = encoding.matmul(direction) # (num, 1)
    target_projection = target_projection.unsqueeze(-1)
    encoding = encoding + (target_projection - current_projection) * direction.transpose(0, 1)
    return encoding