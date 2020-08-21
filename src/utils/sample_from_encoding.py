import torch
from src.utils.convert_tensor_to_texts import convert_tensor_to_texts

def sample_from_encoding(model, vocab, encoding, batch_size):
    sample_num = encoding.size(1)
    sentences = []
    start = 0
    model.eval()

    with torch.no_grad():
        while start < sample_num:
            end = min(sample_num, start + batch_size)
            output = model.sample(encoding=encoding[:, start:end, :])
            sentences.extend(convert_tensor_to_texts(output, vocab))
            start = end
    return sentences