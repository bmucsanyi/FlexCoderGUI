from builtins import bool
from dataclasses import dataclass, field
from typing import Union, Optional

from src import utils

# 1st: multiple input samples of singular arrays
# 2nd: single input sample of a singular array
InputType = Union[list[list[int]], list[int]]
# 1st: multiple output samples of singular arrays
# 2nd: single output sample of a singular array
#      or multiple output samples of single numbers
# 3rd: single output sample of a single number
OutputType = Union[list[list[int]], list[int], int]

InputWithIndicesType = Union[list[list[tuple[int, int]]], list[tuple[int, int]]]
OutputWithIndicesType = Union[
    list[list[tuple[int, int]]], list[tuple[int, int]], tuple[int, int]
]


@dataclass
class Input:
    _data: Optional[InputType] = None
    id: int = field(default_factory=utils.generator)

    @property
    def data(self) -> Optional[InputType]:
        """Returns the underlying ``_data`` of ``self``."""
        return self._data

    @data.setter
    def data(self, value) -> None:
        """Sets the underlying ``_data`` of ``self`` to ``value``.

        Args:
            value: New value of ``_data``.

        Raises:
            ValueError:
                Not a non-empty list was provided.
        """
        if not isinstance(value, list) or not value or any(x == [] for x in value):
            raise ValueError("Non-empty list is needed.")
        self._data = value

    def is_more_io(self) -> bool:
        """Returns whether ``self``'s ``_data`` field contains multiple
        input-output examples.
        """
        return Input.is_more_io_data(self.data)

    @staticmethod
    def is_more_io_data(data: Union[InputType, InputWithIndicesType]) -> bool:
        """Returns whether ``data`` contains multiple input-output examples."""
        return bool(data) and isinstance(data[0], list)

    def __eq__(self, other):
        return type(other) is type(self) and self.id == other.id

    def __str__(self):
        return str(self.data)
