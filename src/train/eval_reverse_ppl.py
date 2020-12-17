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
import math
from src.module.criterion.language_cross_entropy import LanguageCrossEntropyLoss
from src.utils.generate_pad import generate_pad

def eval_reverse_ppl(config: dict, sample_path: str = None) -> float:
	
	os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

	logging.basicConfig(level=logging.DEBUG,
						format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
	logger = logging.getLogger(__name__)

	base_path = config["base_path"]
	vocab_path = os.path.join(base_path, "vocab.pkl")
	embedding_path = os.path.join(base_path, "embedding.npy")
	save_path = os.path.join(base_path, "reverse_ppl_language_model.pkl")

	with open(vocab_path, "rb") as handle:
		vocab = pickle.load(handle)

	if sample_path == None:
	
		reverse_ppl_sample_path = os.path.join(base_path, "reverse_ppl_sample.tsv")
		vae_path = os.path.join(base_path, "text_vae.pkl")

		vae = torch.load(vae_path)

		reverse_ppl_sample_num = config["vanilla_sample"]["sample_num"]

		batch_size = config["text_vae"]["training"]["batch_size"]

		batch_sizes = [batch_size] * (reverse_ppl_sample_num // batch_size) + ([reverse_ppl_sample_num % batch_size] if reverse_ppl_sample_num % batch_size != 0 else [])

		sentences = ["sentence"]

		logger.info("sample")

		for batch_size in batch_sizes:
			output = vae.sample(num=batch_size)
			sentences.extend(convert_tensor_to_texts(output, vocab))

		sentences = [[sentence] for sentence in sentences]

		with open(reverse_ppl_sample_path, "w") as f:
			writer = csv.writer(f, delimiter="\t")
			writer.writerows(sentences)

	else:

		reverse_ppl_sample_path = sample_path
	
	TEXT = data.Field(sequential=True, lower=True, batch_first=True, init_token=SOS, eos_token=EOS)
	fields = [("sentence", TEXT)]

	config = config["language_model"]

	train_path = reverse_ppl_sample_path
	dev_path = os.path.join(base_path, "dev.tsv")
	test_path = os.path.join(base_path, "test.tsv")

	train_data = TabularDataset(path=train_path, format="tsv", skip_header=True, fields=fields)
	dev_data = TabularDataset(path=dev_path, format="tsv", skip_header=True, fields=fields)
	test_data = TabularDataset(path=test_path, format="tsv", skip_header=True, fields=fields)
	TEXT.vocab = vocab
	vocab_size = len(vocab.itos)
	
	device = torch.device("cuda:0")
	train_iter = Iterator(train_data, batch_size=config["training"]["batch_size"], shuffle=True, device=device)
	dev_iter = Iterator(dev_data, batch_size=config["training"]["batch_size"], shuffle=False, device=device)
	test_iter = Iterator(test_data, batch_size=config["training"]["batch_size"], shuffle=False, device=device)

	model = LanguageModel(
		vocab_size=vocab_size,
		embed_size=config["model"]["embed_size"],
		hidden_size=config["model"]["hidden_size"],
		num_layers=config["model"]["num_layers"],
		dropout=config["model"]["dropout"],
		weight_tying=config["model"]["weight_tying"]
	)
	model.load_pretrained_embeddings(path=embedding_path)

	model = model.to(device)

	criterion = LanguageCrossEntropyLoss(ignore_index=PAD_INDEX)
	optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])

	corr_dev_nll = 1e9
	corr_dev_ppl = 1e9
	patience = 0
	max_patience = 20

	for epoch in range(config["training"]["epoches"]):

		for i, batch in enumerate(train_iter):

			model.train()
			optimizer.zero_grad()

			sentence = batch.sentence
			input_sentence = sentence
			batch_size = sentence.size(0)
			pad = generate_pad(size=(batch_size, 1), device=sentence.device)
			output_sentence = torch.cat((sentence[:, 1:], pad), dim=-1)

			logit = model(input_sentence)
			nll, seq_lens = criterion(logit, output_sentence)

			loss = nll.mean()
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), config["training"]["clip_grad_norm"])
			optimizer.step()

			if i % config["training"]["eval_freq"] == 0:
				dev_nll, dev_ppl = eval_language_model(model, dev_iter, criterion)

				if dev_ppl < corr_dev_ppl:
					corr_dev_nll = dev_nll
					corr_dev_ppl = dev_ppl
					torch.save(model, save_path)
					patience = 0
				else:
					patience += 1

				if patience == max_patience:
					break

		if patience == max_patience:
			break

	model = torch.load(save_path)
	test_loss, test_ppl = eval_language_model(model, test_iter, criterion)

	os.remove(save_path)
	return test_ppl