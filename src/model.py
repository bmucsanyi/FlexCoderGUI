from typing import Iterable, Optional

import pytorch_lightning as pl
import torch

# noinspection PyPep8Naming
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split

import config
from src import grammar
from src import utils
from src.dataset import FlexDataset


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_GRU = nn.GRU(
            input_size=1, hidden_size=256, num_layers=2, batch_first=True
        )
        self.output_GRU = nn.GRU(
            input_size=1, hidden_size=256, num_layers=2, batch_first=True
        )

    def forward(
        self,
        input_: torch.Tensor,
        input_lengths: list[list[int]],
        output: torch.Tensor,
        output_lengths: list[int],
    ) -> torch.Tensor:
        embeddings = []

        for i in range(config.STATE_TUPLE_LENGTH_HIB):
            packed_input = nn.utils.rnn.pack_padded_sequence(
                input_[:, i, ...],
                input_lengths[i],
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden_x = self.input_GRU(packed_input)
            embeddings.append(hidden_x[1])  # Embedding of second input GRU

        packed_output = nn.utils.rnn.pack_padded_sequence(
            output, output_lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden_y = self.output_GRU(packed_output)
        embeddings.append(hidden_y[1])  # Embedding of second output GRU

        return torch.cat(embeddings, dim=-1)


def init_lecun_normal(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=(1 / m.in_features) ** 0.5)


class FunctionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.intermediate_layers = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1024),
            nn.SELU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.SELU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.SELU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.SELU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.SELU(),
            nn.Linear(in_features=512, out_features=256),
            nn.SELU(),
            nn.Linear(in_features=256, out_features=128),
            nn.SELU(),
        )
        self.intermediate_layers.apply(init_lecun_normal)
        self.output_layer = nn.Linear(
            in_features=128, out_features=len(grammar.DEFINITIONS) + 1
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.intermediate_layers(input_))


class IndexBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.intermediate_layers = nn.Sequential(
            nn.Linear(
                in_features=1024 + len(grammar.DEFINITIONS) + 1, out_features=512
            ),
            nn.SELU(),
            nn.Linear(in_features=512, out_features=512),
            nn.SELU(),
            nn.Linear(in_features=512, out_features=512),
            nn.SELU(),
            nn.Linear(in_features=512, out_features=256),
            nn.SELU(),
            nn.Linear(in_features=256, out_features=128),
            nn.SELU(),
        )
        self.intermediate_layers.apply(init_lecun_normal)
        self.output_layer = nn.Linear(
            in_features=128, out_features=config.STATE_TUPLE_LENGTH_HIB
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.intermediate_layers(input_))


class BoolLambdaBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.intermediate_layers = nn.Sequential(
            nn.Linear(
                in_features=1024 + len(grammar.DEFINITIONS) + 1, out_features=512
            ),
            nn.SELU(),
            nn.Linear(in_features=512, out_features=512),
            nn.SELU(),
            nn.Linear(in_features=512, out_features=512),
            nn.SELU(),
            nn.Linear(in_features=512, out_features=256),
            nn.SELU(),
            nn.Linear(in_features=256, out_features=128),
            nn.SELU(),
        )
        self.intermediate_layers.apply(init_lecun_normal)
        self.operator_layer = nn.Linear(
            in_features=128, out_features=len(grammar.BOOL_LAMBDA_OPERATORS) + 1
        )
        self.number_layer = nn.Linear(
            in_features=128, out_features=len(grammar.BOOL_LAMBDA_NUMBERS) + 1
        )

    def forward(self, input_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.intermediate_layers(input_)
        return self.operator_layer(x), self.number_layer(x)


class NumLambdaBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.intermediate_layers = nn.Sequential(
            nn.Linear(
                in_features=1024 + len(grammar.DEFINITIONS) + 1, out_features=512
            ),
            nn.SELU(),
            nn.Linear(in_features=512, out_features=512),
            nn.SELU(),
            nn.Linear(in_features=512, out_features=512),
            nn.SELU(),
            nn.Linear(in_features=512, out_features=256),
            nn.SELU(),
            nn.Linear(in_features=256, out_features=128),
            nn.SELU(),
        )
        self.intermediate_layers.apply(init_lecun_normal)
        self.operator_layer = nn.Linear(
            in_features=128, out_features=len(grammar.NUM_LAMBDA_OPERATORS) + 1
        )
        self.number_layer = nn.Linear(
            in_features=128, out_features=len(grammar.NUM_LAMBDA_NUMBERS) + 1
        )

    def forward(self, input_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.intermediate_layers(input_)
        return self.operator_layer(x), self.number_layer(x)


class NumBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.intermediate_layers = nn.Sequential(
            nn.Linear(
                in_features=1024 + len(grammar.DEFINITIONS) + 1, out_features=512
            ),
            nn.SELU(),
            nn.Linear(in_features=512, out_features=512),
            nn.SELU(),
            nn.Linear(in_features=512, out_features=512),
            nn.SELU(),
            nn.Linear(in_features=512, out_features=256),
            nn.SELU(),
            nn.Linear(in_features=256, out_features=128),
            nn.SELU(),
        )
        self.intermediate_layers.apply(init_lecun_normal)
        self.output_layer = nn.Linear(
            in_features=128, out_features=len(grammar.TAKE_DROP_NUMBERS) + 1
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.intermediate_layers(input_))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.function_block = FunctionBlock()
        self.index_block = IndexBlock()
        self.bool_lambda_block = BoolLambdaBlock()
        self.num_lambda_block = NumLambdaBlock()
        self.num_block = NumBlock()

    def forward(self, input_: torch.Tensor) -> tuple:
        function_out = self.function_block(input_)
        block_input = torch.cat([input_, function_out], dim=-1)
        index_output = self.index_block(block_input)
        bool_lambda_output = self.bool_lambda_block(block_input)
        num_lambda_output = self.num_lambda_block(block_input)
        num_output = self.num_block(block_input)

        return (
            function_out,
            index_output,
            *bool_lambda_output,
            *num_lambda_output,
            num_output,
        )


class FlexNet(pl.LightningModule):
    def __init__(self, filename: Optional[str] = None, batch_size: int = 4096):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.val_accuracy = pl.metrics.Accuracy()
        self.train_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

        self.batch_size = batch_size
        self.learning_rate = 1e-3

        if filename is not None:
            self.add_datasets(filename)
        else:
            self.train_ds = None
            self.val_ds = None
            self.test_ds = None

    def add_datasets(self, filename: str) -> None:
        ds = FlexDataset(filename)
        len_ds = len(ds)
        len_train = int(0.97 * len_ds)
        len_val = int(0.02 * len_ds)
        len_test = len_ds - len_train - len_val

        self.train_ds, self.val_ds, self.test_ds = random_split(
            ds, [len_train, len_val, len_test]
        )

    def forward(
        self,
        input_: torch.Tensor,
        input_lengths: list[list[int]],
        output: torch.Tensor,
        output_lengths: list[int],
    ) -> tuple:
        embedding = self.encoder(input_, input_lengths, output, output_lengths)
        return self.decoder(embedding)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss, prediction_label_zip = self.calculate_loss_acc(batch)
        self.log_loss_acc(loss, prediction_label_zip, "train")
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        loss, prediction_label_zip = self.calculate_loss_acc(batch)
        self.log_loss_acc(loss, prediction_label_zip, "val")

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        loss, prediction_label_zip = self.calculate_loss_acc(batch)
        self.log_loss_acc(loss, prediction_label_zip, "test")

    def log_loss_acc(
        self, loss: torch.Tensor, prediction_label_zip: Iterable, log_string: str
    ) -> None:
        self.log(f"{log_string}_loss", loss)
        for i, (out, label) in prediction_label_zip:
            if i == 1:
                acc = self.val_accuracy(torch.sigmoid(out), label)  # indices
            else:
                acc = self.val_accuracy(F.softmax(out, dim=1), label)
            self.log(f"{log_string}_acc_{i}", acc)

    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def calculate_loss_acc(self, batch: tuple) -> tuple:
        (
            ((batch_input, input_lengths), (batch_output, output_lengths)),
            batch_labels,
        ) = batch
        prediction = self(batch_input, input_lengths, batch_output, output_lengths)

        loss = sum(
            F.binary_cross_entropy_with_logits(out, label.float())
            if i == 1
            else F.cross_entropy(out, label)
            for i, (out, label) in enumerate(zip(prediction, batch_labels))
        )
        return loss, enumerate(zip(prediction, batch_labels))

    def train_dataloader(self):
        if self.train_ds is None:
            raise ValueError("No dataset provided.")

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=utils.collate_fn,
        )

    def val_dataloader(self):
        if self.val_ds is None:
            raise ValueError("No dataset provided.")

        return DataLoader(
            self.val_ds, batch_size=self.batch_size, collate_fn=utils.collate_fn
        )

    def test_dataloader(self):
        if self.test_ds is None:
            raise ValueError("No dataset provided.")

        return DataLoader(
            self.test_ds, batch_size=self.batch_size, collate_fn=utils.collate_fn
        )
