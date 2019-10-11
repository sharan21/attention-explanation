import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim

from typing import *
from pathlib import Path
from allennlp.data.dataset_readers import LanguageModelingReader
from allennlp.data.tokenizers import CharacterTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.training import Trainer
from sklearn.model_selection import train_test_split
from torch import nn

from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.models import Model
from model.custom_lstm import OptimizedLSTM
from allennlp.nn.util import get_text_field_mask
from model.custom_lstm import NaiveLSTM, OptimizedLSTM

DATA_ROOT = Path("../data/brown")


N_EPOCHS = 1

char_tokenizer = CharacterTokenizer(lowercase_characters=True)

reader = LanguageModelingReader(
    tokens_per_instance=500,
    tokenizer=char_tokenizer,
    token_indexers = {"tokens": SingleIdTokenIndexer()},
)

train_ds = reader.read(DATA_ROOT / "brown.txt")
train_ds, val_ds = train_test_split(train_ds, random_state=0, test_size=0.1)

vocab = Vocabulary.from_instances(train_ds)

iterator = BasicIterator(batch_size=32)
iterator.index_with(vocab)

def train(model: nn.Module, epochs: int=10):
    trainer = Trainer(
        model=model.cuda() if torch.cuda.is_available() else model,
        optimizer=optim.Adam(model.parameters()),
        iterator=iterator, train_dataset=train_ds,
        validation_dataset=val_ds, num_epochs=epochs,
        cuda_device=0 if torch.cuda.is_available() else -1
    )
    return trainer.train()


class LanguageModel(Model):

    def __init__(self, encoder: nn.RNN, vocab: Vocabulary,
                 embedding_dim: int = 50):
        super().__init__(vocab=vocab)
        # char embedding
        self.vocab_size = vocab.get_vocab_size()
        self.padding_idx = vocab.get_token_index("@@PADDING@@")
        token_embedding = Embedding(
            num_embeddings=vocab.get_vocab_size(),
            embedding_dim=embedding_dim,
            padding_index=self.padding_idx,
        )
        self.embedding = BasicTextFieldEmbedder({"tokens": token_embedding})
        self.encoder = encoder
        self.projection = nn.Linear(self.encoder.hidden_size, self.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)

    def forward(self, input_tokens: Dict[str, torch.Tensor],output_tokens: Dict[str, torch.Tensor]):

        embs = self.embedding(input_tokens)
        x, _ = self.encoder(embs)
        x = self.projection(x)
        if output_tokens is not None:
            loss = self.loss(x.view((-1, self.vocab_size)), output_tokens["tokens"].flatten())
        else:
            loss = None
        return {"loss": loss, "logits": x}

if __name__ == "__main__":
    lm_naive = LanguageModel(NaiveLSTM(50, 125), vocab)
    train(lm_naive, epochs=N_EPOCHS)
