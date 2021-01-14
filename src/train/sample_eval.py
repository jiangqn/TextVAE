import os
import torch
from torch import nn
import pickle
import csv
from src.utils.convert_tensor_to_texts import convert_tensor_to_texts
from src.get_features.get_ppl import get_ppl_from_tsv
from src.constants import PAD_INDEX

def sample_eval_by_language_model(model, base_path, sample_num=1000, batch_size=64, **kwargs):

    vocab_path = os.path.join(base_path, "vocab.pkl")
    language_model_path = os.path.join(base_path, "language_model.pkl")
    sample_save_path = os.path.join(base_path, "sample_eval.tsv")

    with open(vocab_path, "rb") as handle:
        vocab = pickle.load(handle)

    batch_sizes = [batch_size] * (sample_num // batch_size) + [sample_num % batch_size]

    sentences = ["sentence"]

    for batch_size in batch_sizes:
        output = model.sample(num=batch_size, max_len=kwargs["max_len"])
        sentences.extend(convert_tensor_to_texts(output, vocab))

    sentences = [[sentence] for sentence in sentences]

    with open(sample_save_path, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(sentences)

    language_model = kwargs.get("language_model", None)
    if language_model == None:
        ppls = get_ppl_from_tsv(file_path=sample_save_path, batch_size=batch_size, model_path=language_model_path, vocab=vocab)
    else:
        ppls = get_ppl_from_tsv(file_path=sample_save_path, batch_size=batch_size, model=language_model, vocab=vocab)
    ppl = sum(ppls) / len(ppls)
    os.remove(sample_save_path)
    return ppl

def sample_eval_by_vae(model, sample_num=10000, batch_size=64):

    batch_sizes = [batch_size] * (sample_num // batch_size) + [sample_num % batch_size]
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)

    total_tokens = 0
    total_ce_loss = 0

    model.train()
    with torch.no_grad():
        for batch_size in batch_sizes:
            batch_output, batch_logit = model.sample(num=batch_size, output_logit=True)
            output_size = batch_logit.size(-1)
            batch_output = batch_output.view(-1)
            batch_logit = batch_logit.view(-1, output_size)
            loss = criterion(batch_logit, batch_output)
            mask = (batch_output != PAD_INDEX)
            token_num = mask.long().sum().item()
            total_tokens += token_num
            total_ce_loss += loss.item() * token_num

    ce_loss = total_ce_loss / total_tokens
    ppl = 2 ** ce_loss
    return ppl