import json
from typing import Union

import torch

# noinspection PyPep8Naming
import torch.nn.functional as F
from torch.utils.data import Dataset

import config
from src import grammar
from src.input_ import InputType


class FlexDataset(Dataset):
    def __init__(self, filename: str):
        raw_data = FlexDataset.load_data(filename)

        (
            self.inputs,
            self.input_lengths,
            self.outputs,
            self.output_lengths,
            self.definitions,
            self.indices,
            self.bool_lambda_ops,
            self.bool_lambda_nums,
            self.num_lambda_ops,
            self.num_lambda_nums,
            self.take_drop_nums,
        ) = self.process_raw_data(raw_data)

    @staticmethod
    def load_data(filename: str) -> list:
        raw_data_list = []
        with open(filename) as f:
            for line in f:
                raw_data_list.append(json.loads(line))

        return raw_data_list

    @staticmethod
    def process_raw_data(raw_data: list[dict]) -> tuple:
        inputs = []
        input_lengths = []
        outputs = []
        output_lengths = []
        definitions = []
        indices = []
        bool_lambda_ops = []
        bool_lambda_nums = []
        num_lambda_ops = []
        num_lambda_nums = []
        take_drop_nums = []

        for ind, sample in enumerate(raw_data):
            processed_input, input_length = FlexDataset.to_processed_tensor(
                sample["input"], is_input=True
            )
            inputs.append(processed_input)
            input_lengths.append(input_length)

            processed_output, output_length = FlexDataset.to_processed_tensor(
                sample["output"], is_input=False
            )
            outputs.append(processed_output)
            output_lengths.append(output_length)

            function_dict = sample["next_transformation"]["function"]

            processed_definition = FlexDataset.process_scalar(
                function_dict["definition"], grammar.DEFINITIONS
            )
            definitions.append(processed_definition)

            processed_indices = FlexDataset.process_indices(
                sample["next_transformation"]["indices"]
            )
            indices.append(processed_indices)

            processed_bool_lambda_op = FlexDataset.process_scalar(
                function_dict["bool_lambda_operator"], grammar.BOOL_LAMBDA_OPERATORS
            )
            bool_lambda_ops.append(processed_bool_lambda_op)

            processed_bool_lambda_num = FlexDataset.process_scalar(
                function_dict["bool_lambda_number"], grammar.BOOL_LAMBDA_NUMBERS
            )
            bool_lambda_nums.append(processed_bool_lambda_num)

            processed_num_lambda_op = FlexDataset.process_scalar(
                function_dict["num_lambda_operator"], grammar.NUM_LAMBDA_OPERATORS
            )
            num_lambda_ops.append(processed_num_lambda_op)

            processed_num_lambda_num = FlexDataset.process_scalar(
                function_dict["num_lambda_number"], grammar.NUM_LAMBDA_NUMBERS
            )
            num_lambda_nums.append(processed_num_lambda_num)

            processed_take_drop_num = FlexDataset.process_scalar(
                function_dict["take_drop_number"], grammar.TAKE_DROP_NUMBERS
            )
            take_drop_nums.append(processed_take_drop_num)

            del raw_data[ind]

        return (
            torch.stack(inputs),
            input_lengths,
            torch.stack(outputs),
            output_lengths,
            torch.tensor(definitions),
            torch.stack(indices),
            torch.tensor(bool_lambda_ops),
            torch.tensor(bool_lambda_nums),
            torch.tensor(num_lambda_ops),
            torch.tensor(num_lambda_nums),
            torch.tensor(take_drop_nums),
        )

    @staticmethod
    def to_processed_tensor(
            data: list[InputType], is_input: bool
    ) -> tuple[torch.Tensor, Union[int, list[int]]]:
        def convert_and_pad(
                list_, max_length=config.INPUT_LENGTH_HIB, pad_value=config.PAD_VALUE
        ) -> tuple[torch.Tensor, int]:
            tensor = torch.tensor(list_, dtype=torch.float32)
            original_length = tensor.shape[0]
            padded_tensor = F.pad(
                tensor, (0, max_length - original_length), value=pad_value
            )
            padded_tensor = padded_tensor.view(-1, 1)

            return padded_tensor, original_length

        if is_input:
            ret = []
            length_ret = []
            for elem in data:
                processed_elem, length = convert_and_pad(elem)
                ret.append(processed_elem)
                length_ret.append(length)

            ret.extend(
                [
                    torch.full((config.INPUT_LENGTH_HIB, 1), config.MASKING_VALUE)
                    for _ in range(config.STATE_TUPLE_LENGTH_HIB - len(ret))
                ]
            )

            length_ret.extend(
                [
                    config.INPUT_LENGTH_HIB
                    for _ in range(config.STATE_TUPLE_LENGTH_HIB - len(length_ret))
                ]
            )

            return torch.stack(ret), length_ret
        else:
            return convert_and_pad(data[0])

    @staticmethod
    def process_indices(indices: list[int]) -> torch.Tensor:
        ret = torch.zeros(config.STATE_TUPLE_LENGTH_HIB, dtype=torch.long)
        for index in indices:
            ret[index] = 1

        return ret

    @staticmethod
    def process_scalar(data: str, data_list: list[str]) -> int:
        if data in data_list:
            return data_list.index(data)
        else:
            return len(data_list)  # new dimension: 'does not contain'

    def __getitem__(self, index: int) -> tuple:
        return (
            (
                (self.inputs[index], self.input_lengths[index]),
                (self.outputs[index], self.output_lengths[index]),
            ),
            (
                self.definitions[index],
                self.indices[index],
                self.bool_lambda_ops[index],
                self.bool_lambda_nums[index],
                self.num_lambda_ops[index],
                self.num_lambda_nums[index],
                self.take_drop_nums[index],
            ),
        )

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __str__(self) -> str:
        return (
            f"{self.inputs.shape=}\n"
            f"{len(self.input_lengths)=}\n"
            f"{self.outputs.shape=}\n"
            f"{len(self.output_lengths)=}\n"
            f"{self.definitions.shape=}\n"
            f"{self.indices.shape=}\n"
            f"{self.bool_lambda_ops.shape=}\n"
            f"{self.bool_lambda_nums.shape=}\n"
            f"{self.num_lambda_ops.shape=}\n"
            f"{self.num_lambda_nums.shape=}\n"
            f"{self.take_drop_nums.shape=}"
        )
