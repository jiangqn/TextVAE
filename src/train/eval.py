import torch
from torch import nn
from src.constants import PAD_INDEX
from src.train.sample_eval import sample_eval_by_language_model
from src.utils.generate_pad import generate_pad
import math

def eval_text_cnn(model, data_iter, criterion=None):

    total_samples = 0
    correct_samples = 0
    total_loss = 0

    model.eval()
    with torch.no_grad():

        for batch in data_iter:

            sentence = batch.sentence
            label = batch.label

            logit = model(sentence)
            prediction = logit.argmax(dim=-1)
            batch_size = label.size(0)
            total_samples += batch_size
            correct_samples += (prediction == label).long().sum().item()

            if criterion != None:
                loss = criterion(logit, label)
                total_loss += loss.item() * batch_size

    accuracy = correct_samples / total_samples
    if criterion != None:
        loss = total_loss / total_samples
        return loss, accuracy
    else:
        return accuracy

def eval_language_model(model, data_iter, criterion):

    total_samples = 0
    total_tokens = 0
    total_nll = 0

    model.eval()
    with torch.no_grad():

        for batch in data_iter:

            sentence = batch.sentence
            input_sentence = sentence
            batch_size = sentence.size(0)
            pad = generate_pad(size=(batch_size, 1), device=sentence.device)
            output_sentence = torch.cat((sentence[:, 1:], pad), dim=-1)

            logit = model(input_sentence)
            nll, seq_lens = criterion(logit, output_sentence)

            total_samples += batch_size
            total_tokens += seq_lens.long().sum().item()
            total_nll += nll.sum().item()

    nll = total_nll / total_samples
    ppl = math.exp(total_nll / total_tokens)

    return nll, ppl

def eval_text_vae(model, data_iter, criterion, kldiv, base_path, **kwargs):

    total_samples = 0
    total_tokens = 0

    total_reconstruction = 0
    total_kl = 0
    total_nll = 0
    correct_tokens = 0

    model.eval()

    with torch.no_grad():

        for batch in data_iter:

            sentence = batch.sentence
            src = sentence[:, 1:]
            trg_input = sentence
            batch_size = sentence.size(0)
            pad = generate_pad(size=(batch_size, 1), device=sentence.device)
            trg_output = torch.cat((sentence[:, 1:], pad), dim=-1)

            logit, posterior_mean, posterior_std = model(src, trg_input)

            reconstruction, seq_lens = criterion(logit, trg_output)
            kl = kldiv(posterior_mean, posterior_std)
            nll = reconstruction + kl
            ppl = torch.exp(nll / seq_lens)

            total_samples += batch_size
            total_tokens += seq_lens.long().sum().item()

            total_reconstruction += reconstruction.sum().item()
            total_kl += kl.sum().item()
            total_nll += nll.sum().item()

            mask = (trg_output != PAD_INDEX)
            prediction = logit.argmax(dim=-1)
            correct_tokens += (prediction.masked_select(mask) == trg_output.masked_select(mask)).long().sum().item()

    reconstruction = total_reconstruction / total_samples
    kl = total_kl / total_samples
    nll = total_nll / total_samples
    ppl = math.exp(total_nll / total_tokens)
    wer = 1 - correct_tokens / total_tokens
    forward_ppl = sample_eval_by_language_model(model, base_path, **kwargs)

    return reconstruction, kl, nll, ppl, wer, forward_ppl