import torch
from src.constants import PAD_INDEX

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

    total_tokens = 0
    total_loss = 0

    model.eval()
    with torch.no_grad():

        for batch in data_iter:

            sentence = batch.sentence
            input_sentence = sentence
            batch_size = sentence.size(0)
            pad = torch.zeros(size=(batch_size, 1), dtype=torch.long, device=sentence.device)
            output_sentence = torch.cat((sentence[:, 1:], pad), dim=-1)

            logit = model(input_sentence)
            output_sentence = output_sentence.view(-1)
            output_size = logit.size(-1)
            logit = logit.view(-1, output_size)
            loss = criterion(logit, output_sentence)

            mask = (output_sentence != PAD_INDEX)
            token_num = mask.long().sum().item()
            total_tokens += token_num
            total_loss += token_num * loss.item()

    loss = total_loss / total_tokens
    return loss

def eval_text_vae(model, data_iter, criterion):

    total_tokens = 0
    total_loss = 0
    correct_tokens = 0

    model.eval()

    with torch.no_grad():

        for batch in data_iter:

            sentence = batch.sentence
            src = sentence[:, 1:]
            trg_input = sentence
            batch_size = sentence.size(0)
            pad = torch.zeros(size=(batch_size, 1), dtype=torch.long, device=sentence.device)
            trg_output = torch.cat((sentence[:, 1:], pad), dim=-1)

            logit, _, _ = model(src, trg_input)
            trg_output = trg_output.view(-1)
            output_size = logit.size(-1)
            logit = logit.view(-1, output_size)
            loss = criterion(logit, trg_output)

            mask = (trg_output != PAD_INDEX)
            token_num = mask.long().sum().item()
            total_tokens += token_num
            total_loss += token_num * loss.item()
            prediction = logit.argmax(dim=-1)
            correct_tokens += (prediction.masked_select(mask) == trg_output.masked_select(mask)).long().sum().item()

    loss = total_loss / total_tokens
    wer = 1 - correct_tokens / total_tokens
    return loss, wer