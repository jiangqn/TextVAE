import torch
from typing import List, Tuple
from collections import Counter
from src.utils.multinomial_distribution import sample_from_multinomial_distribution

def rejection_sample(num_layers: int, num: int, hidden_size: int, classifier, label: int) -> torch.Tensor:

    encoding_bucket = []
    current_num = 0
    batch_size = 100

    while current_num < num:
        encoding = torch.randn(size=(num_layers, batch_size, hidden_size), dtype=torch.float32)
        transformed_encoding = encoding.transpose(0, 1).reshape(batch_size, num_layers * hidden_size).cpu().numpy()
        prediction, confidience = classifier.predict(transformed_encoding, output_probability=True)
        prediction = torch.from_numpy(prediction).long()
        confidience = torch.from_numpy(confidience).float()
        index = torch.where((prediction == label) & (confidience >= 0.95))[0]
        encoding = encoding.index_select(dim=1, index=index)
        valid_num = min(num - current_num, encoding.size(1))
        encoding_bucket.append(encoding[:, 0:valid_num:, :])
        current_num += valid_num

    encoding = torch.cat(encoding_bucket, dim=1)
    return encoding

def multinomial_rejection_sample(num_layers: int, sample_num: int, hidden_size: int, classifier, distribution: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    :return sample: torch.LongTensor (sample_num,)
    :return encoding: torch.FloatTensor (num_layers, sample_num, hidden_size)
    '''

    sample = sample_from_multinomial_distribution(distribution, sample_num)
    counter = Counter(sample)

    encoding = []
    sample = []
    for label, num in counter.items():
        if num > 0:
            encoding.append(rejection_sample(num_layers, num, hidden_size, classifier, label))
            sample.extend([label] * num)
    encoding = torch.cat(encoding, dim=1)
    sample = torch.tensor(sample)
    index = torch.randperm(sample_num)
    encoding = encoding[:, index, :]
    sample = sample[index]
    return sample, encoding