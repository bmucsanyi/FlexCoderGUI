import copy
import operator
import random
from dataclasses import dataclass, field
from typing import Optional

from src.func_impl import *
from src.grammar import (
    FUNC_DICT,
    OPERATOR_DICT,
    REVERSE_OPERATOR_DICT,
    PARSER,
    ALL_FUNCTION_STRINGS,
)
from src.input_ import (
    Input,
    InputType,
    OutputType,
    InputWithIndicesType,
    OutputWithIndicesType,
)


@dataclass
class Function:
    definition: Callable
    operator: Optional[Callable]
    number: Optional[int]
    inputs: list[Input] = field(default_factory=list)

    @classmethod
    def from_dict(cls, func_dict: dict) -> "Function":
        """Factory method that creates a Function instance from a dictionary
        object containing the parts of the function as strings.

        The dictionary object is required to have a ``"function"`` key. Optional
        keys are ``"operator"`` and ``"num"``. All other keys are discarded.
        """
        function = func_dict["function"]
        function = FUNC_DICT[function]

        if "operator" in func_dict:
            operator_ = func_dict["operator"]
            operator_ = OPERATOR_DICT[operator_]
        else:
            operator_ = None

        if "number" in func_dict:
            number = func_dict["number"]
            number = int(number)
        else:
            number = None

        return cls(definition=function, operator=operator_, number=number)

    @classmethod
    def from_detailed_dict(cls, func_dict: dict) -> "Function":
        """Factory method that creates a Function instance from a dictionary
        object containing the parts of the function as strings.

        The dictionary object is required to have a ``"function"`` key. Optional
        keys are ``"operator"`` and ``"num"``. All other keys are discarded.
        """
        function = func_dict["definition"]
        function = FUNC_DICT[function]

        if func_dict["num_lambda_operator"] is not None:
            operator_ = func_dict["num_lambda_operator"]
            operator_ = OPERATOR_DICT[operator_]
        elif func_dict["bool_lambda_operator"] is not None:
            operator_ = func_dict["bool_lambda_operator"]
            operator_ = OPERATOR_DICT[operator_]
        else:
            operator_ = None

        if func_dict["take_drop_number"] is not None:
            number = func_dict["take_drop_number"]
            number = int(number)
        elif func_dict["num_lambda_number"] is not None:
            number = func_dict["num_lambda_number"]
            number = int(number)
        elif func_dict["bool_lambda_number"] is not None:
            number = func_dict["bool_lambda_number"]
            number = int(number)
        else:
            number = None

        return cls(definition=function, operator=operator_, number=number)

    def eval(self, param_inputs: Optional[list[InputType]] = None) -> OutputType:
        """Evaluates ``self`` using its ``inputs`` attribute or the
        ``param_inputs`` parameter.
        The method overrides the ``inputs`` attribute of ``self`` if
        ``param_inputs`` is not None. If ``self``'s ``definition`` is zip_with,
        it is possible to only have one half of the input specified in its
        ``inputs`` attribute, as the other half might came from another
        function in a composition. In such case, the ``inputs`` attribute of
        the function is combined with the ``param_inputs`` parameter, producing
        the entire input to the function.

        Args:
            param_inputs: An optional input list to the function. If provided,
              it must be of length one or two if the function is zip-with,
              otherwise length one.
        Returns:
            An output of type OutputType. If there are multiple IO examples
              specified in the array attribute or parameter, the output
              will be a list of output values corresponding to each
              example.
        Raises:
            ValueError: Invalid combination of inputs are provided, or the
              examples are mismatching in the number of input-output examples.
        """
        if param_inputs is None:
            param_inputs = []

        if self.definition is zip_with and len(self.inputs) == 2 and not param_inputs:
            used_inputs = [input_.data for input_ in self.inputs]
        elif (
            self.definition is zip_with
            and len(self.inputs) == 1
            and len(param_inputs) == 1
        ):
            used_inputs = [param_inputs[0], self.inputs[0].data]
        elif self.definition is zip_with and not self.inputs and len(param_inputs) == 2:
            used_inputs = param_inputs
        elif (
            self.definition is not zip_with
            and len(self.inputs) == 1
            and not param_inputs
        ):
            used_inputs = [self.inputs[0].data]
        elif (
            self.definition is not zip_with
            and not self.inputs
            and len(param_inputs) == 1
        ):
            used_inputs = param_inputs
        else:
            raise ValueError("Invalid combination of inputs provided.")

        if self.definition is zip_with:
            if (
                Input.is_more_io_data(used_inputs[0])
                != Input.is_more_io_data(used_inputs[1])
                or Input.is_more_io_data(used_inputs[0])
                and Input.is_more_io_data(used_inputs[1])
                and len(used_inputs[0]) != len(used_inputs[1])
            ):
                raise ValueError("Mismatching I/O examples provided.")

            def lambda_func(x, y):
                return self.operator(x, y)

            if not Input.is_more_io_data(used_inputs[0]):
                return self.definition(lambda_func, *used_inputs)
            else:
                return [
                    self.definition(lambda_func, a, b) for a, b in zip(*used_inputs)
                ]
        elif self.definition in {take, drop}:
            function_args = (self.number,)
        elif self.definition in {map_func, filter_func}:
            if self.operator is operator.mod and self.definition is filter_func:

                def lambda_func(x):
                    return self.operator(x, self.number) == 0

            else:

                def lambda_func(x):
                    return self.operator(x, self.number)

            function_args = (lambda_func,)
        else:
            function_args = ()

        if not Input.is_more_io_data(used_inputs[0]):
            return self.definition(*function_args, used_inputs[0])
        else:
            return [self.definition(*function_args, a) for a in used_inputs[0]]

    def eval_with_indices(
        self,
        param_inputs: list[InputWithIndicesType] = None,
        id_1: Optional[int] = None,
        id_2: Optional[int] = None,
    ) -> tuple[OutputWithIndicesType, int]:
        """Evaluates ``self`` using its ``inputs`` attribute or the
        ``param_inputs`` parameter, returning the output and the corresponding
        original indices in one of the inputs, as well as the _id of the
        aforementioned input.

        The method overrides the ``inputs`` attribute of ``self`` if
        ``param_inputs`` is not None. If ``self``'s ``definition`` is zip_with,
        it is possible to only have one half of the input specified in its
        ``inputs`` attribute, as the other half might came from another
        function in a composition. In such case, the ``inputs`` attribute of
        the function is combined with the ``param_inputs`` parameter, producing
        the entire input to the function. If ``self`` is zip_with, then the
        output indices and _id will be randomly chosen between the inputs.


        Args:
            param_inputs: An optional input list to the function, already
              annotated with indices. If provided, it must be of length one or
              two if the function is zip-with, otherwise length one.
            id_1: Id of one of the left branch's inputs the composition object
              containing the function instance has.
            id_2: Id of one of the right branch's inputs the composition object
              containing the function instance has.

        Returns:
            An output of type ``OutputWithIndicesType``. If there are multiple IO examples
              specified in the array attribute or parameter, the output
              will be a list of output values corresponding to each
              example, annotated with indices of one of the inputs and its _id.

        Raises:
            ValueError: Invalid combination of inputs are provided, or the
              examples are mismatching in the number of input-output examples.
        """

        def add_indices(input_):
            for sample in input_:
                for i, elem in enumerate(sample):
                    sample[i] = (elem, i)

        def check_io_match(input_a, input_b):
            a_more_io = Input.is_more_io_data(input_a)
            b_more_io = Input.is_more_io_data(input_b)
            if (
                a_more_io != b_more_io
                or a_more_io
                and b_more_io
                and len(input_a) != len(input_b)
            ):
                raise ValueError("Mismatching I/O examples provided.")

        if id_1 is None:
            id_1 = self.inputs[0].id

        if len(self.inputs) == 2:
            id_2 = self.inputs[1].id
        elif self.inputs and self.definition is zip_with:
            id_2 = self.inputs[0].id

        if id_2 is None:
            ret_id = id_1
        else:
            ret_id = random.choice([id_1, id_2])

        if param_inputs is None:
            param_inputs = []

        if self.definition is zip_with and len(self.inputs) == 2 and not param_inputs:
            check_io_match(self.inputs[0].data, self.inputs[1].data)

            input_1 = copy.deepcopy(self.inputs[0].data)
            input_2 = copy.deepcopy(self.inputs[1].data)
            if self.inputs[0].is_more_io():
                add_indices(input_1)
                add_indices(input_2)
            else:
                add_indices([input_1])
                add_indices([input_2])

            used_inputs = input_1, input_2
        elif (
            self.definition is zip_with
            and len(self.inputs) == 1
            and len(param_inputs) == 1
        ):
            check_io_match(self.inputs[0].data, param_inputs[0])

            input_2 = copy.deepcopy(self.inputs[0].data)
            if self.inputs[0].is_more_io():
                add_indices(input_2)
            else:
                add_indices([input_2])
            used_inputs = [param_inputs[0], input_2]
        elif self.definition is zip_with and not self.inputs and len(param_inputs) == 2:
            check_io_match(param_inputs[0], param_inputs[1])

            used_inputs = param_inputs
        elif (
            self.definition is not zip_with
            and len(self.inputs) == 1
            and not param_inputs
        ):
            input_1 = copy.deepcopy(self.inputs[0].data)
            if self.inputs[0].is_more_io():
                add_indices(input_1)
            else:
                add_indices([input_1])
            used_inputs = [input_1]
        elif (
            self.definition is not zip_with
            and not self.inputs
            and len(param_inputs) == 1
        ):
            used_inputs = param_inputs
        else:
            raise ValueError("Invalid combination of inputs provided.")

        if self.definition is zip_with:

            def lambda_func(x, y):
                elem = x[1] if ret_id == id_1 else y[1]
                return self.operator(x[0], y[0]), elem

            if not Input.is_more_io_data(used_inputs[0]):
                return (
                    self.definition(lambda_func, used_inputs[0], used_inputs[1]),
                    ret_id,
                )
            else:
                return (
                    [
                        self.definition(lambda_func, a, b)
                        for a, b in zip(used_inputs[0], used_inputs[1])
                    ],
                    ret_id,
                )
        elif self.definition in {take, drop}:
            function_args = (self.number,)
        elif self.definition in {map_func, filter_func}:
            if self.operator is operator.mod and self.definition is filter_func:

                def lambda_func(x):
                    return self.operator(x[0], self.number) == 0

            elif self.definition is filter_func:

                def lambda_func(x):
                    return self.operator(x[0], self.number)

            else:  # self.definition is map_func

                def lambda_func(x):
                    return self.operator(x[0], self.number), x[1]

            function_args = (lambda_func,)
        else:
            function_args = ()

        if not Input.is_more_io_data(used_inputs[0]):
            if self.definition in (sum, min, max, length):
                temp = [t[0] for t in used_inputs[0]]
                return self.definition(temp), ret_id

            return self.definition(*function_args, used_inputs[0]), ret_id
        else:
            if self.definition in (sum, min, max, length):
                return (
                    [self.definition(b[0] for b in a) for a in used_inputs[0]],
                    ret_id,
                )

            return [self.definition(*function_args, a) for a in used_inputs[0]], ret_id

    def as_dict(self) -> dict:
        """Returns a dictionary representation of ``self``.

        The resulting dict contains the keys ``"definition"``,
        ``"num_lambda_number"``, ``"num_lambda_operator"``,
        ``"bool_lambda_number"``, ``"bool_lambda_operator"`` and
        ``"take_drop_number"``.
        """
        definition = self.definition.__name__

        if self.definition is map_func:
            num_lambda_number = str(self.number)
            num_lambda_operator = REVERSE_OPERATOR_DICT[self.operator]
        elif self.definition is zip_with:
            num_lambda_number = None
            num_lambda_operator = REVERSE_OPERATOR_DICT[self.operator]
        else:
            num_lambda_number = num_lambda_operator = None

        if self.definition is filter_func:
            bool_lambda_number = str(self.number)
            bool_lambda_operator = REVERSE_OPERATOR_DICT[self.operator]
        else:
            bool_lambda_number = bool_lambda_operator = None

        if self.definition in {take, drop}:
            take_drop_number = str(self.number)
        else:
            take_drop_number = None

        return {
            "definition": definition,
            "num_lambda_number": num_lambda_number,
            "num_lambda_operator": num_lambda_operator,
            "bool_lambda_number": bool_lambda_number,
            "bool_lambda_operator": bool_lambda_operator,
            "take_drop_number": take_drop_number,
        }

    def is_evaluable(self) -> bool:
        """Returns whether the function is evaluable with its provided inputs."""
        if self.definition is zip_with:
            return len(self.inputs) == 2
        else:
            return len(self.inputs) == 1

    def __str__(self) -> str:
        func_name = self.definition.__name__

        if self.definition is zip_with:
            operator_string = REVERSE_OPERATOR_DICT[self.operator]

            if not self.inputs:
                input_1, input_2 = "input", "input"
            elif len(self.inputs) == 1:
                input_1, input_2 = "input", str(self.inputs[0])
            else:
                input_1, input_2 = str(self.inputs[0]), str(self.inputs[1])

            if operator_string in ("min", "max"):
                param_string = f"{operator_string}, {input_1}, {input_2}"
            else:
                param_string = (
                    f"lambda x, y: x {operator_string} y, {input_1}, {input_2}"
                )
        elif self.definition in {map_func, filter_func}:
            operator_string = REVERSE_OPERATOR_DICT[self.operator]
            input_string = "input" if not self.inputs else str(self.inputs[0])

            if self.operator is operator.mod and self.definition is filter_func:
                param_string = (
                    f"lambda x: x {operator_string} {self.number} == 0, {input_string}"
                )
            else:
                param_string = (
                    f"lambda x: x {operator_string} {self.number}, {input_string}"
                )
        elif self.definition in {take, drop}:
            input_string = "input" if not self.inputs else str(self.inputs[0])
            param_string = f"{self.number}, {input_string}"
        else:
            param_string = "input" if not self.inputs else str(self.inputs[0])

        return f"{func_name}({param_string})"

    def is_order_altering(self):
        non_order_altering_funcs = (take, drop, filter_func, map_func)
        order_altering_func = self.definition not in non_order_altering_funcs

        is_order_altering_map = (
            self.definition is map_func
            and (self.operator is operator.mul and self.number < 0)
            or self.operator is operator.mod
        )
        # Note: map(-n,...) could also be order altering but that
        # is rare and can't be decided without the arrays

        return order_altering_func or is_order_altering_map

    def eval_on_state_tuple(self, state_tuple: tuple, indices):

        if self.definition is copy_state_tuple:
            try:
                return self.definition(state_tuple, indices[0])
            except FullStateTupleError:
                return None

        elif self.definition is not zip_with:
            try:
                res = tuple(
                    [self.eval(inp)] if ind in indices else inp
                    for ind, inp in enumerate(state_tuple)
                )

                return res
            except Exception:
                return None

        else:
            temp = list(state_tuple)
            inp1 = temp[indices[0]]
            inp2 = temp[indices[1]]

            for ind in sorted(indices, reverse=True):
                temp.pop(ind)

            try:
                res = self.eval([inp1[0], inp2[0]])
            except:
                return None

            temp.append([res])
            return tuple(temp)

    @staticmethod
    def get_all_functions() -> list["Function"]:
        """Returns all possible unique Function instances derivable from the
        grammar defined in ``src.grammar.py``.
        """
        return [
            Function.from_dict(PARSER(function)) for function in ALL_FUNCTION_STRINGS
        ]

    @staticmethod
    def get_buckets() -> dict[Callable, list["Function"]]:
        """Returns buckets for each function type. Buckets contain all
        possible parameterizations of the corresponding function type.
        """
        all_function_names = FUNC_DICT.values()
        all_functions = Function.get_all_functions()
        return {
            name: [
                function for function in all_functions if function.definition is name
            ]
            for name in all_function_names
        }

    @staticmethod
    def get_array_buckets() -> dict[Callable, list["Function"]]:
        """Returns buckets for each function type that return an array. Buckets
        contain all possible parameterizations of the corresponding function type.
        """
        all_buckets = Function.get_buckets()
        for name in {min, max, sum, length}:
            del all_buckets[name]
        return all_buckets


ALL_FUNCTIONS = Function.get_all_functions()

BUCKETS = Function.get_buckets()


ARRAY_BUCKETS = Function.get_array_buckets()


def main():
    map_ = Function.from_dict({"function": "map_func", "operator": "*", "number": "2"})
    map_.inputs = [Input([9, 6, 7, 3, 4])]
    print(map_)

    take_ = Function.from_dict({"function": "take", "number": "2"})
    print(take_)

    reverse = Function.from_dict({"function": "reverse_func"})
    reverse.inputs = [Input([1, 5, 3, 2, 9])]
    print(reverse)

    zw1 = Function.from_dict({"function": "zip_with", "operator": "*"})
    zw1.inputs = [Input([9, 4, 9, 4, 3])]
    print(zw1)

    filter_ = Function.from_dict(
        {"function": "filter_func", "operator": ">", "number": "7"}
    )
    print(filter_)

    zw2 = Function.from_dict({"function": "zip_with", "operator": "+"})
    print(zw2)

    func = Function.from_dict({"function": "zip_with", "operator": "*"})
    func.inputs = [Input([1, 2, 3])]
    print(func.eval_with_indices([[(5, 3), (6, 0)]], id_1=123))

    func = Function.from_dict({"function": "drop", "number": "2"})
    func.inputs = [Input([1, 2, 3])]
    print(func.eval_with_indices())


if __name__ == "__main__":
    main()
