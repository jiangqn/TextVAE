from src.model.text_vae import TextVAE

from src.module.encoder.encoder import Encoder
from src.module.encoder.bow_encoder import BOWEncoder
from src.module.encoder.bow_mlp_encoder import BOWMLPEncoder
from src.module.encoder.conv_encoder import ConvEncoder
from src.module.encoder.gru_encoder import GRUEncoder
from src.module.encoder.lstm_encoder import LSTMEncoder

from src.module.decoder.decoder import Decoder
from src.module.decoder.gru_decoder import GRUDecoder
from src.module.decoder.lstm_decoder import LSTMDecoder
from src.module.decoder.skip_gru_decoder import SkipGRUDecoder
from src.module.decoder.skip_lstm_decoder import SkipLSTMDecoder

def build_encoder(encoder_config: dict) -> Encoder:
    vocab_size = encoder_config["vocab_size"]
    encoder_type = encoder_config["encoder_type"]
    encoder_config = encoder_config[encoder_type]
    if encoder_type == "bow_encoder":
        encoder = BOWEncoder(
            vocab_size=vocab_size,
            embed_size=encoder_config["embed_size"],
            dropout=encoder_config["dropout"]
        )
    elif encoder_type == "bow_mlp_encoder":
        encoder = BOWMLPEncoder(
            vocab_size=vocab_size,
            embed_size=encoder_config["embed_size"],
            hidden_size=encoder_config["hidden_size"],
            dropout=encoder_config["dropout"]
        )
    elif encoder_type == "conv_encoder":
        encoder = ConvEncoder(
            vocab_size=vocab_size,
            embed_size=encoder_config["embed_size"],
            kernel_sizes=encoder_config["kernel_sizes"],
            kernel_num=encoder_config["kernel_num"],
            num_layers=encoder_config["num_layers"],
            dropout=encoder_config["dropout"]
        )
    elif encoder_type == "gru_encoder":
        encoder = GRUEncoder(
            vocab_size=vocab_size,
            embed_size=encoder_config["embed_size"],
            hidden_size=encoder_config["hidden_size"],
            num_layers=encoder_config["num_layers"],
            bidirectional=encoder_config["bidirectional"],
            dropout=encoder_config["dropout"],
            output_type=encoder_config["output_type"]
        )
    elif encoder_type == "lstm_encoder":
        encoder = LSTMEncoder(
            vocab_size=vocab_size,
            embed_size=encoder_config["embed_size"],
            hidden_size=encoder_config["hidden_size"],
            num_layers=encoder_config["num_layers"],
            bidirectional=encoder_config["bidirectional"],
            dropout=encoder_config["dropout"],
            output_type=encoder_config["output_type"]
        )
    else:
        raise ValueError("encoder_type: %s is error." % encoder_type)
    return encoder

def build_decoder(decoder_config: dict) -> Decoder:
    vocab_size = decoder_config["vocab_size"]
    latent_size = decoder_config["latent_size"]
    decoder_type = decoder_config["decoder_type"]
    decoder_config = decoder_config[decoder_type]
    if decoder_type == "gru_decoder":
        decoder = GRUDecoder(
            vocab_size=vocab_size,
            embed_size=decoder_config["embed_size"],
            hidden_size=decoder_config["hidden_size"],
            latent_size=latent_size,
            num_layers=decoder_config["num_layers"],
            dropout=decoder_config["dropout"],
            word_dropout=decoder_config["word_dropout"],
            decoder_generator_tying=decoder_config["decoder_generator_tying"],
            initial_hidden_type=decoder_config["initial_hidden_type"]
        )
    elif decoder_type == "lstm_decoder":
        decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embed_size=decoder_config["embed_size"],
            hidden_size=decoder_config["hidden_size"],
            latent_size=latent_size,
            num_layers=decoder_config["num_layers"],
            dropout=decoder_config["dropout"],
            word_dropout=decoder_config["word_dropout"],
            decoder_generator_tying=decoder_config["decoder_generator_tying"],
            initial_hidden_type=decoder_config["initial_hidden_type"]
        )
    elif decoder_type == "skip_gru_decoder":
        decoder = SkipGRUDecoder(
            vocab_size=vocab_size,
            embed_size=decoder_config["embed_size"],
            hidden_size=decoder_config["hidden_size"],
            latent_size=latent_size,
            num_layers=decoder_config["num_layers"],
            dropout=decoder_config["dropout"],
            word_dropout=decoder_config["word_dropout"],
            decoder_generator_tying=decoder_config["decoder_generator_tying"],
            initial_hidden_type=decoder_config["initial_hidden_type"]
        )
    elif decoder_type == "skip_lstm_decoder":
        decoder = SkipLSTMDecoder(
            vocab_size=vocab_size,
            embed_size=decoder_config["embed_size"],
            hidden_size=decoder_config["hidden_size"],
            latent_size=latent_size,
            num_layers=decoder_config["num_layers"],
            dropout=decoder_config["dropout"],
            word_dropout=decoder_config["word_dropout"],
            decoder_generator_tying=decoder_config["decoder_generator_tying"],
            initial_hidden_type=decoder_config["initial_hidden_type"]
        )
    else:
        raise ValueError("decoder_type: %s is error." % decoder_type)
    return decoder

def build_model(config: dict) -> TextVAE:

    encoder_config = config["encoder"]
    encoder_config["vocab_size"] = config["vocab_size"]
    encoder = build_encoder(encoder_config)

    decoder_config = config["decoder"]
    decoder_config["vocab_size"] = config["vocab_size"]
    decoder_config["latent_size"] = config["latent_size"]
    decoder = build_decoder(decoder_config)

    model = TextVAE(
        encoder=encoder,
        decoder=decoder,
        latent_size=config["latent_size"],
        encoder_decoder_tying=config["encoder_decoder_tying"]
    )

    return model