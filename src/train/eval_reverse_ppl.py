import torch
from torch import nn
from torch import optim
from torchtext import data
from torchtext.data import TabularDataset, Iterator
import os
import pickle
import csv
import logging
from src.model.language_model import LanguageModel
from src.utils.convert_tensor_to_texts import convert_tensor_to_texts
from src.constants import SOS, EOS, PAD_INDEX
from src.train.eval import eval_language_model

def eval_reverse_ppl(config: dict, sample_path: str = None) -> float:
	
	os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

	logging.basicConfig(level=logging.DEBUG,
						format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
	logger = logging.getLogger(__name__)

	base_path = config['base_path']
	vocab_path = os.path.join(base_path, 'vocab.pkl')
	embedding_path = os.path.join(base_path, 'embedding.npy')
	save_path = os.path.join(base_path, 'reverse_ppl_language_model.pkl')

	with open(vocab_path, 'rb') as handle:
		vocab = pickle.load(handle)

	if sample_path == None:
	
		reverse_ppl_sample_path = os.path.join(base_path, 'reverse_ppl_sample.tsv')
		vae_path = os.path.join(base_path, 'vae.pkl')

		vae = torch.load(vae_path)

		# reverse_ppl_sample_num = config['vanilla_sample']['sample_num']

		reverse_ppl_sample_num = 200000

		batch_size = config['vae']['batch_size']

		batch_sizes = [batch_size] * (reverse_ppl_sample_num // batch_size) + ([reverse_ppl_sample_num % batch_size] if reverse_ppl_sample_num % batch_size != 0 else [])

		sentences = ['sentence']

		logger.info('sample')

		for batch_size in batch_sizes:
			output = vae.sample(num=batch_size)
			sentences.extend(convert_tensor_to_texts(output, vocab))

		sentences = [[sentence] for sentence in sentences]

		with open(reverse_ppl_sample_path, 'w') as f:
			writer = csv.writer(f, delimiter='\t')
			writer.writerows(sentences)

	else:

		reverse_ppl_sample_path = sample_path
	
	TEXT = data.Field(sequential=True, lower=True, batch_first=True, init_token=SOS, eos_token=EOS)
	fields = [('sentence', TEXT)]

	config = config['language_model']

	train_path = reverse_ppl_sample_path
	dev_path = os.path.join(base_path, 'dev.tsv')
	test_path = os.path.join(base_path, 'test.tsv')

	train_data = TabularDataset(path=train_path, format='tsv', skip_header=True, fields=fields)
	dev_data = TabularDataset(path=dev_path, format='tsv', skip_header=True, fields=fields)
	test_data = TabularDataset(path=test_path, format='tsv', skip_header=True, fields=fields)
	TEXT.vocab = vocab
	vocab_size = len(vocab.itos)
	
	device = torch.device('cuda:0')
	train_iter = Iterator(train_data, batch_size=config['batch_size'], shuffle=True, device=device)
	dev_iter = Iterator(dev_data, batch_size=config['batch_size'], shuffle=False, device=device)
	test_iter = Iterator(test_data, batch_size=config['batch_size'], shuffle=False, device=device)

	logger.info('build model')
	model = LanguageModel(
		vocab_size=vocab_size,
		embed_size=config['embed_size'],
		hidden_size=config['hidden_size'],
		num_layers=config['num_layers'],
		dropout=config['dropout'],
		weight_tying=config['weight_tying']
	)
	model.load_pretrained_embeddings(path=embedding_path)

	logger.info('transfer model to GPU')
	model = model.to(device)

	logger.info('set up criterion and optimizer')
	criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
	optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

	logger.info('start train')

	min_dev_loss = 1e9
	patience = 0
	max_patience = 20

	for epoch in range(config['epoches']):

		total_tokens = 0
		total_loss = 0

		for i, batch in enumerate(train_iter):

			model.train()
			optimizer.zero_grad()

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
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
			optimizer.step()

			mask = (output_sentence != PAD_INDEX)
			token_num = mask.long().sum().item()
			total_tokens += token_num
			total_loss += token_num * loss.item()

			if i % config['eval_freq'] == 0:
				train_loss = total_loss / total_tokens
				total_loss = 0
				total_tokens = 0
				dev_loss = eval_language_model(model, dev_iter, criterion)
				# logger.info('[epoch %2d step %4d]\ttrain_loss: %.4f train_ppl: %.4f dev_loss: %.4f dev_ppl: %.4f' %
				# 			(epoch, i, train_loss, 2 ** train_loss, dev_loss, 2 ** dev_loss))

				if dev_loss < min_dev_loss:
					min_dev_loss = dev_loss
					torch.save(model, save_path)
					patience = 0
				else:
					patience += 1

				if patience == max_patience:
					break

		if patience == max_patience:
			break

	logger.info('dev_loss: %.4f\tdev_ppl: %.4f' % (min_dev_loss, 2 ** min_dev_loss))

	model = torch.load(save_path)
	test_loss = eval_language_model(model, test_iter, criterion)
	logger.info('test_loss: %.4f\ttest_ppl: %.4f' % (test_loss, 2 ** test_loss))

	os.remove(reverse_ppl_sample_path)
	os.remove(save_path)

	logger.info('finish')

	test_ppl = 2 ** test_loss
	return test_ppl