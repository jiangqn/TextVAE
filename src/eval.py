import torch

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