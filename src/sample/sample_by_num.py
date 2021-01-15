import torch
from torch import nn
from torchtext.vocab import Vocab
from typing import List, Tuple
from src.utils.convert_tensor_to_texts import convert_tensor_to_texts
from src.get_features.get_length import get_length
import numpy as np

def sample_by_num(model: nn.Module, vocab: Vocab, num: int, max_len: int, batch_size: int = 64) -> Tuple[List[str], np.ndarray]:
    """
    :param model: text_vae nn.Module
    :param vocab: Vocab
    :param num: int
    :param batch_size: int
    :return sentences: List[str]
    """

    sentences = []
    latent_variables = []

    model.eval()

    with torch.no_grad():

        count = 0

        while count < num:

            output, output_latent_variable = model.sample(num=batch_size, max_len=max_len + 1, output_latent_variable=True)
            output_sentences = convert_tensor_to_texts(output, vocab)
            selected_sentences, selected_latent_variable = select(output_sentences, output_latent_variable, max_len)
            selected_count = selected_latent_variable.shape[0]
            selected_count = min(selected_count, num - count)
            selected_sentences = selected_sentences[0: selected_count]
            selected_latent_variable = selected_latent_variable[0: selected_count]
            sentences.extend(selected_sentences)
            latent_variables.append(selected_latent_variable)

            count += selected_count

    latent_variables = np.concatenate(latent_variables, axis=0)
    return sentences, latent_variables

def select(sentences: List[str], latent_variable: np.ndarray, max_len: int) -> Tuple[List[str], torch.Tensor]:

    """
    :param sentences: List[str]
    :param latent_variable: torch.FloatTensor (num, latent_size)
    :param max_len: int
    :return selected_sentences: List[str]
    :return selected_latent_variable: torch.FloatTensor
    """

    assert len(sentences) == latent_variable.shape[0]

    sentences_length = np.asarray(get_length(sentences))
    index = np.where(sentences_length <= max_len)[0]
    selected_sentences = [sentences[i] for i in index]
    selected_latent_variable = latent_variable[index]
    assert len(selected_sentences) == selected_latent_variable.shape[0]
    return selected_sentences, selected_latent_variable