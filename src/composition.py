import collections
import copy
import operator
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import reduce
from typing import Deque, Optional, Union, Generator, Iterable

import numpy as np

from src import grammar
from src import utils
from src.func_impl import *
from src.function import Function
from src.input_ import Input, InputType, OutputType, OutputWithIndicesType

OPERATOR_COMBINATIONS = {(operator.sub, operator.add), (operator.add, operator.sub)}


class UnimprovableInputError(Exception):
    """A Composition object's current inputs cannot be improved so as to
    stay in the desired output boundary provided in ``config.py``."""


class UnevaluableCompositionError(Exception):
    """A Composition object cannot be evaluated, as not all the necessary
    inputs are provided.
    """


class UncorrectableSampleError(Exception):
    """A Composition object consisting of a single drop function returns an
    empty list as output and no function can be deleted, as the result would
    be an empty composition.
    """


class UnsatisfiableRangeConstraintError(Exception):
    """A Composition object cannot be provided any inputs with the current
    output range constraint in ``config.py``."""


class UnreachableCodeError(Exception):
    """Exception subclass that denotes code that is (or should) be unreachable."""


@dataclass
class Composition:
    _root_function: Function
    _children: list["Composition"] = field(default_factory=list)
    _parent: Optional["Composition"] = None
    _id: int = field(default_factory=utils.generator)

    @property
    def root_function(self):
        return self._root_function

    @property
    def children(self):
        return self._children

    @property
    def parent(self):
        return self._parent

    @property
    def id(self):
        return self._id

    @classmethod
    def from_composition(
        cls,
        function: Function,
        branch_1: "Composition",
        branch_2: Optional["Composition"] = None,
    ) -> "Composition":
        """Returns a new Composition object with ``function`` applied to the
        top of ``branch_1`` and optionally ``branch_2``.
        Args:
            function: A Function object that is to be applied to the top of
              the other provided Composition objects.
            branch_1: Mandatory Composition object. Parameter ``function``
              extends it further. If ``function`` is a zip_with instance,
              the left child of the resulting composition's root will be
              ``branch_1``. Otherwise, ``branch_1`` will be its only child.
            branch_2: Optional Composition object. It can only be provided
              if ``function`` is a zip_with instance. In that case,
              ``branch_2`` will be the resulting composition's root's right
              child.
        Returns:
            A new Composition object whose _root_function will be ``function``,
            and the root's _children will be ``branch_1``, and ``branch_2`` if
            provided.
        Raises:
            ValueError: A valid Composition object cannot be constructed from
              the input parameter as outlined in this docstring.
        """
        if function is None or branch_1 is None:
            raise ValueError("None provided as parameter.")
        elif function.definition is not zip_with and branch_2 is not None:
            raise ValueError("Function provided expects a single list as input.")

        function = copy.deepcopy(function)
        branch_1 = copy.deepcopy(branch_1)
        branch_2 = copy.deepcopy(branch_2)

        if function.definition is zip_with:
            return cls._from_branches(function, branch_1, branch_2)
        else:
            return cls._from_branch(function, branch_1)

    def eval(self) -> OutputType:
        """Evaluates ``self`` using the inputs of its leaf nodes' functions."""
        return Composition._postorder_eval(self)

    def eval_with_indices(self) -> tuple[OutputWithIndicesType, int]:
        """Evaluates ``self`` using the inputs of its leaf nodes' functions,
        and also returns one of the inputs' original index positions, as well
        as the _id of the aforementioned input."""
        return Composition._postorder_eval_with_indices(self)

    @property
    def leaves(self) -> list["Composition"]:
        """Provides the leaves of ``self``.
        Returns:
            A list of leaf nodes present in ``self`` from left to right.
            If some leaves are copied, they are present in the list as many
            times as they are in ``self``.
        """
        leaves = []
        self._preorder_leaves(self, leaves)
        return leaves

    @property
    def leaf_inputs(self) -> list[Input]:
        """Provides the inputs of the leaves of ``self``.
        Returns:
            A list of Input objects present in ``self``'s leaves from
            left to right. If some inputs or leaves are copied, they are
            present in the list as many times as they are in ``self``.
        """
        return [input_ for leaf in self.leaves for input_ in leaf._root_function.inputs]

    @property
    def num_inputs(self) -> int:
        """Provides the number of inputs the composition has."""
        return self._preorder_num_inputs(self)

    def is_more_io(self) -> bool:
        for node in self:
            if (
                not node.children
                or len(node.children) == 1
                and node._root_function.definition is zip_with
            ):
                return node._root_function.inputs[0].is_more_io()

    def as_samples(self, num_io_examples: Optional[int] = None) -> Optional[list[dict]]:
        """Provides ``self``'s representation as samples that are used
        to train the neural network used in the synthesizing process.
        ``self``'s inputs must be correctly filled before calling this
        method.

        Args:
            num_io_examples (int): The number of input-output examples
              in the samples. If not provided, ``self`` is expected to
              have been correctly filled with inputs previously.
              Otherwise the samples are filled with random inputs that
              make the output of the composition stay in the boundary
              provided in ``config.py``.

        Returns:
            A list of dictionary objects with keys ``"input"``, ``"output"``,
            and ``"next_transformation"``. ``"input"`` is the current state
            tuple, ``"output"`` is the goal state tuple, and
            ``"next_transformation"`` is a dictionary representation of the
            function that takes the current state tuple one step closer to the
            goal state tuple, with the keys ``"definition"``, ``"number"`` and
            ``"operator"``.
        """
        copy_comp = copy.deepcopy(self)

        if num_io_examples is not None:
            copy_comp = Composition.fill_random_inputs(
                copy_comp, num_io_examples, check_output_boundary=True
            )
        else:
            num_io_examples = copy_comp.is_more_io()

        copy_comp = Composition._optimize_drop_filter(copy_comp)

        unique_inputs = []
        for input_ in copy_comp.leaf_inputs:
            if input_ not in unique_inputs:
                unique_inputs.append(input_)
        random.shuffle(unique_inputs)
        state_tuple = tuple(unique_inputs)

        samples = []
        left_index_dict = {}

        output = copy_comp.eval()
        if isinstance(output, int):
            output = [output]
        elif num_io_examples > 1 and isinstance(output[0], int):
            output = [[elem] for elem in output]

        copy_comp._postorder_sample_generation(
            copy_comp,
            state_tuple,
            output,
            samples,
            copy_comp.copy_ids,
            left_index_dict,
        )

        return samples

    def subcompositions(self, length_: int) -> list["Composition"]:
        """Provides subcompositions of length ``length_`` of ``self``.
        Args:
            length\\_: The length of the subcompositions that are to be returned.
        Returns:
            A list consisting of every subcomposition of length ``length_`` of
            ``self``.
        """
        subcomps = []
        self._preorder_subcompositions(self, length_, subcomps)
        return subcomps

    def extend_leaf(
        self, leaf: "Composition", sub_comp: "Composition", index: int = None
    ) -> None:
        """Extends the specified leaf ``leaf`` of ``self`` with
        ``sub_comp`` in-place.
        Args:
            leaf: A leaf subcomposition of self.
            sub_comp: A Composition object that is to be attached to ``leaf``
              as a child.
            index: The index where sub_comp will be put in ``leaf``'s _children
              list.
        Raises:
            ValueError:
                ``leaf`` is a copied branch, thus it cannot be extended, as it
                would break the invariant of copied branches.
        """
        if leaf._id in self.copy_ids:
            raise ValueError("Only non-copy branches are allowed to be extended")

        new_sub_comp = copy.deepcopy(sub_comp)
        leaf._add_child(new_sub_comp, index=index)
        leaf._root_function.inputs.clear()

    def extend_leaves(
        self, roots_to_add: list[Optional["Composition"]], indices: list[Optional[int]]
    ) -> None:
        """Extends the leaves  of ``self`` with ``roots_to_add`` in-place.
        Args:
            roots_to_add: List consisting of either Composition objects or
              None values. None values are to be provided if the corresponding
              leaf of ``self`` is not to be extended (ordered from left
              to right). Otherwise, the composition must be provided with which
              the corresponding leaf of ``self`` is to be extended.
            indices: List consisting of either integers or None values. None
              values are to be provided if either the corresponding leaf of
              ``self`` is not to be extended or the default index for the
              extension is desired. Otherwise, the index must be provided
              on which the corresponding leaf of ``self`` is to be extended.
        Raises:
            ValueError:
                The length of the provided lists do not match.
                ``roots_to_add`` must be as long as many _children ``self``
                can accept. Indices must be as long as ``roots_to_add``.
        """

        # This produces a list where every leaf is
        # present as many times as it has inputs
        flattened_leaves = []
        for leaf in self.leaves:
            if leaf._root_function.definition is zip_with and leaf.children:
                flattened_leaves.append(leaf)
            else:
                for _ in range(leaf.num_inputs):
                    flattened_leaves.append(leaf)

        if not (len(roots_to_add) == len(flattened_leaves) == len(indices)):
            raise ValueError(
                "Provide a Composition and index for each of the leaves. "
                "Use None for the leaves you wish to attach nothing to!"
            )

        for leaf, comp_to_add, index in zip(flattened_leaves, roots_to_add, indices):
            if comp_to_add is not None:
                self.extend_leaf(leaf=leaf, sub_comp=comp_to_add, index=index)

    @property
    def copy_ids(self) -> list[int]:
        """Provides the ids of the subcompositions or inputs in ``self`` that
        are copies.
        A subcomposition or input is a copy of another in ``self`` iff their
        ids match.
        """
        ret_dict = defaultdict(int)
        Composition._postorder_copy_ids(self, ret_dict)
        return [key for key in ret_dict for _ in range(ret_dict[key] - 1)]

    def copy_with_new_ids(self) -> "Composition":
        """Provides a recursive copy of ``self`` with different ids than the
        ones in ``self``.
        Returns:
            A new Composition object with the same subcompositions and inputs
              as ``self`` has, but new ids.
        """
        # TODO: handle copies
        new_comp = self._preorder_deepcopy(comp=self, parent=None, new_ids=True)

        return new_comp

    def postorder_iterate(self) -> Generator["Composition", None, None]:
        for child in self.children:
            yield from child.postorder_iterate()

        yield self

    def fill_input_by_id(self, new_data: InputType, input_id: int) -> None:
        for leaf in self.leaves:
            for inp in leaf._root_function.inputs:
                if inp.id == input_id:
                    inp.data = new_data

    @staticmethod
    def fill_random_inputs(
        comp: "Composition", num_io_examples: int, check_output_boundary: bool = False
    ) -> Optional["Composition"]:
        """Fills ``comp`` with random input lists.
        Copies are filled with the same list. The length of the random lists
        is between ``INPUT_LENGTH_LOB`` and ``INPUT_LENGTH_HIB``, two
        parameters that are present in config.py. The range of the elements
        of the lists is between ``INPUT_LOB`` and ``INPUT_HIB`` that are
        other parameters from config.py.

        Args:
            comp: Composition object whose inputs are to be filled.
            num_io_examples: Number of input-output examples. If only one
              input-output example is desired, the data field of the inputs
              of ``comp`` will be of type list[int]. Otherwise, they will be
              of type list[list[int]].
            check_output_boundary: Whether the output boundaries in ``config.py``
              should be checked when filling the inputs randomly. If this is set,
              ``comp`` will also be optimized for its inputs in order to
              provide well-learnable samples for the neural network.

        Returns:
            A new Composition object from ``comp`` that has its inputs filled.
            If ``check_output_boundary`` is set, then the output composition
            will also be an optimized version of ``comp``.
        """
        input_dict = {}

        # Fill inputs without checking output boundary
        comp._naive_fill_inputs(input_dict, num_io_examples)

        if check_output_boundary:
            comp = Composition._optimize_drop_filter(comp)
            deadlock_1 = 0
            deadlock_2 = 0
            while True:
                done, comp = Composition._bottom_up_random_inputs(comp)

                if done:
                    break

                deadlock_1 += 1

                if deadlock_1 == 50:
                    deadlock_2 += 1
                    deadlock_1 = 0
                    if deadlock_2 == 4:
                        raise UnimprovableInputError("The inputs cannot be improved.")

                    comp._naive_fill_inputs(input_dict, num_io_examples)
                    comp = Composition._optimize_drop_filter(comp)
        return comp

    def __str__(self) -> str:
        return self._postorder_string(self).replace("final", "input")

    def __len__(self) -> int:
        return self._postorder_len(self)

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and self._id == other.id

    def __deepcopy__(self, memo) -> "Composition":
        new_comp = self._preorder_deepcopy(comp=self, parent=None)

        return new_comp

    def __iter__(self) -> Generator["Composition", None, None]:
        yield self

        for child in self.children:
            yield from iter(child)

    def __contains__(self, elem: object):
        if isinstance(elem, Input):
            for sub_comp in self:
                for input_ in sub_comp._root_function.inputs:
                    if id(input_) == id(elem):
                        return True
            return False
        elif isinstance(elem, Composition):
            for sub_comp in self:
                if id(sub_comp) == id(elem):
                    return True
            return False
        else:
            return False

    def _first_map(self) -> Optional["Composition"]:
        return self._preorder_first_map(self)

    def _add_child(self, child: "Composition", index: int = None) -> None:
        def check_len(num: int):
            if len(self.children) == num:
                raise ValueError("Maximum number of _children reached.")

        if self._root_function.definition is zip_with:
            check_len(2)
        else:
            check_len(1)
        child._parent = self

        if index is not None:
            self.children.insert(index, child)
        else:
            self.children.append(child)

    def _get_input(self) -> InputType:
        if self.children:
            return self.children[0].eval()
        else:
            return self._root_function.inputs[0].data

    @classmethod
    def _from_branch(cls, function: Function, branch: "Composition") -> "Composition":
        # TODO: fix sort + sort or sort + reverse or sort + map + sort
        if function.definition in (drop, take) and branch._first_map() is not None:
            return cls._merge_function_inside(function, branch)
        elif function.definition in (drop, take, map_func, sort, reverse_func):
            return cls._branch_compressor(function, branch)
        else:
            return cls._merge_branch(function, branch)

    @classmethod
    def _from_branches(
        cls,
        function: Function,
        branch_1: "Composition",
        branch_2: Optional["Composition"] = None,
    ) -> "Composition":
        new_comp = cls(function)
        new_comp._add_child(branch_1)

        if branch_2 is not None:
            new_comp._add_child(branch_2)

        return new_comp

    @classmethod
    def _merge_function_inside(
        cls, function: Function, branch: "Composition"
    ) -> "Composition":
        first_map = branch._first_map()
        if first_map.children:
            child = branch._first_map().children[0]
            if function.definition is take and child._root_function.definition is take:
                min_param = min(function.number, child._root_function.number)
                child._root_function.number = min_param
                return branch
            elif (
                function.definition is drop and child._root_function.definition is drop
            ):
                remaining_space = grammar.MAX_NUM - child._root_function.number
                if function.number > remaining_space:
                    diff = function.number - remaining_space
                    function.number = diff
                    child._root_function.number = grammar.MAX_NUM
                else:
                    child._root_function.number += function.number
                    return branch

                new_comp = cls(function)
                new_comp._parent = first_map
                first_map._children = [new_comp]
                child._parent = new_comp
                new_comp._children = [child]
        else:
            new_comp = cls(function)
            first_map._add_child(new_comp)
            function_1 = function
            function_2 = first_map._root_function
            function_1.inputs, function_2.inputs = function_2.inputs, function_1.inputs
        return branch

    @classmethod
    def _merge_branch(cls, function: Function, branch: "Composition") -> "Composition":
        new_comp = cls(function)
        new_comp._add_child(branch)

        return new_comp

    @classmethod
    def _branch_compressor(
        cls, function: Function, branch: "Composition"
    ) -> "Composition":
        root: Function = branch._root_function
        if function.definition is take and root.definition is take:
            min_param = min(function.number, root.number)
            root.number = min_param
            return branch
        elif function.definition is drop and root.definition is drop:
            remaining_space = grammar.MAX_NUM - root.number
            if function.number > remaining_space:
                diff = function.number - remaining_space
                function.number = diff
                root.number = grammar.MAX_NUM
            else:
                root.number += function.number
                return branch
        elif function.definition is map_func and root.definition is map_func:
            if function.operator == root.operator:
                if function.operator in {
                    operator.mul,
                    operator.add,
                } and function.operator(root.number, function.number) in range(
                    0, grammar.MAX_NUM + 1
                ):
                    # Map: *, + merge and only the root function remains
                    # 0 is for when we are multiplying by 0
                    root.number = function.operator(function.number, root.number)
                    return branch
                elif (
                    function.operator == operator.sub
                    and function.number + root.number in range(1, grammar.MAX_NUM)
                ):
                    # Map: - merge and only the root function remains
                    root.number += function.number
                    return branch
                else:
                    if function.operator in {operator.add, operator.sub}:
                        remaining_space = grammar.MAX_NUM - root.number
                        function.number -= remaining_space
                        root.number = grammar.MAX_NUM
                    elif function.operator == operator.mul:
                        result = function.number * root.number
                        function.number, root.number = utils.calculate_distribution(
                            result
                        )
            elif (function.operator, root.operator) in OPERATOR_COMBINATIONS:
                # Map: +, - merge
                if function.number != root.number:
                    bigger = max(function, root, key=lambda x: x.number)
                    other = root if bigger is function else function
                    root.number = bigger.number - other.number
                    root.operator = bigger.operator

                return branch
            elif (
                root.operator is operator.mul
                and function.operator is operator.floordiv
                and root.number % function.number == 0
            ):
                # Map: *, // merge, but only in this order

                if root.number // function.number != 1:
                    # Else don't change the number
                    root.number //= function.number

                return branch
        elif function.definition is sort:
            node_funcs = set(node._root_function.definition for node in branch)
            if all(func is not sort for func in node_funcs):
                # If there are no sorts there is nothing to do
                return cls._merge_branch(function, branch)

            if branch._root_function.definition is reverse_func:
                return branch

            for node in branch:
                if node._root_function.definition is not sort:
                    continue

                path = Composition._path_to(node)
                funcs_in_path = [p._root_function for p in path]

                # Filter out explicit sort and reverse stacking
                if (
                    len(funcs_in_path) >= 2
                    and funcs_in_path[0].definition is reverse_func
                    and funcs_in_path[1].definition is sort
                ):
                    return branch

                # Remove the last element as it's ``node``
                funcs_in_path = funcs_in_path[:-1]

                # This is true if any functions in the path alter the ordering of the list elements
                if any(func.is_order_altering() for func in funcs_in_path):
                    return cls._merge_branch(function, branch)

                # If we ever reach this the function should not be added
                return branch

            # No problems -> can merge
            return cls._merge_branch(function, branch)
        elif function.definition is reverse_func:
            node_funcs = set(node._root_function.definition for node in branch)
            if all(func is not reverse_func for func in node_funcs):
                # If there are no reverse funcs there is nothing to do
                return cls._merge_branch(function, branch)

            for node in branch:
                if node._root_function.definition is not reverse_func:
                    continue

                path = Composition._path_to(node)
                funcs_in_path = [p._root_function for p in path]

                # Filter out explicit sort and reverse stacking
                if (
                    len(funcs_in_path) >= 2
                    and funcs_in_path[0].definition is sort
                    and funcs_in_path[1].definition is reverse_func
                ):
                    return branch

                # Remove the last element as it's ``node``
                funcs_in_path = funcs_in_path[:-1]

                # This is true if any functions in the path alter the ordering of the list elements
                if branch._root_function.definition is not reverse_func and any(
                    func.is_order_altering() and func.definition is not reverse_func
                    for func in funcs_in_path
                ):
                    return cls._merge_branch(function, branch)

                # if we ever reach this the function should not be added
                return branch

            # No problems -> can merge
            return cls._merge_branch(function, branch)

        return cls._merge_branch(function, branch)

    @staticmethod
    def _path_to(node: "Composition") -> list["Composition"]:
        out: Deque["Composition"] = collections.deque()
        out.appendleft(node)
        while node.parent is not None:
            node = node.parent
            out.appendleft(node)
        return list(out)

    @classmethod
    def _sort_optimizer(
        cls, function: Function, branch: "Composition"
    ) -> "Composition":
        def is_order_altering_function(func: Function):
            non_order_altering_funcs = (take, drop, filter_func, map_func)
            order_altering_func = func.definition not in non_order_altering_funcs

            is_order_altering_map = (
                func.definition is map_func
                and (func.operator is operator.mul and func.number < 0)
                or func.operator is operator.mod
            )
            # Note: map(-n,, ...) could also be order altering but that
            # is rare and can't be decided without the arrays

            return order_altering_func or is_order_altering_map

        node_funcs = set(node._root_function.definition for node in branch)
        if all(func is not sort for func in node_funcs):
            # If there are no sorts there is nothing to do
            return cls._merge_branch(function, branch)

        if branch._root_function.definition is reverse_func:
            return branch

        for node in branch:
            if node._root_function.definition is not sort:
                continue

            # Remove the last element as it's ``node``
            path = Composition._path_to(node)[:-1]
            funcs_in_path = [p._root_function for p in path]

            # This is true if any functions in the path alter the ordering of the list elements
            if any(is_order_altering_function(func) for func in funcs_in_path):
                return cls._merge_branch(function, branch)

            # If we ever reach this the function should not be added
            return branch

        # No problems -> can merge
        return cls._merge_branch(function, branch)

    @classmethod
    def _reverse_optimizer(
        cls, function: Function, branch: "Composition"
    ) -> "Composition":
        def is_order_altering_function(func: Function):
            non_order_altering_funcs = (take, drop, filter_func, map_func)
            order_altering_func = func.definition not in non_order_altering_funcs

            is_order_altering_map = (
                func.definition is map_func
                and (func.operator is operator.mul and func.number < 0)
                or func.operator is operator.mod
            )
            # Note: map(-n,...) could also be order altering but that
            # is rare and can't be decided without the arrays

            return order_altering_func or is_order_altering_map

        node_funcs = set(node._root_function.definition for node in branch)
        if all(func is not reverse_func for func in node_funcs):
            # If there is no reverse func there is nothing to do
            return cls._merge_branch(function, branch)

        for node in branch:
            if node._root_function.definition is not reverse_func:
                continue

            # Remove the last element as it's ``node``
            path = Composition._path_to(node)[:-1]
            funcs_in_path = [p._root_function for p in path]

            # This is true if any functions in the path alter the ordering of the list elements
            if branch._root_function.definition is not reverse_func and any(
                is_order_altering_function(func) for func in funcs_in_path
            ):
                return cls._merge_branch(function, branch)

            # If we ever reach this the function should not be added
            return branch

        # No problems -> can merge
        return cls._merge_branch(function, branch)

    @classmethod
    def _preorder_deepcopy(
        cls, comp: "Composition", parent: Optional["Composition"], new_ids: bool = False
    ) -> "Composition":
        function = copy.deepcopy(comp._root_function)
        new_comp = cls(function)
        if not new_ids:
            new_comp._id = comp._id
        new_comp._children = comp.children.copy()  # Shallow copy
        new_comp._parent = parent

        for i, child in enumerate(new_comp.children):
            new_comp.children[i] = cls._preorder_deepcopy(child, new_comp, new_ids)

        if new_ids:
            for i in range(len(new_comp._root_function.inputs)):
                new_comp._root_function.inputs[i] = Input()

        return new_comp

    # Recursive staticmethods
    # Note that there is no good solution for hardcoding the
    # class's name inside recursive staticmethods that would retain
    # the same level of readability.

    @staticmethod
    def _postorder_eval(comp: "Composition") -> OutputType:
        root: Function = comp._root_function
        if not root.is_evaluable():
            if root.definition is zip_with:
                try:
                    input_1 = Composition._postorder_eval(comp.children[0])
                except IndexError:
                    raise UnevaluableCompositionError(
                        "Not all required arrays are provided."
                    )  # from None

                if root.inputs:
                    return root.eval([input_1])
                else:
                    try:
                        input_2 = Composition._postorder_eval(comp.children[1])
                    except IndexError:
                        raise UnevaluableCompositionError(
                            "Not all required arrays are provided."
                        )  # from None

                    return root.eval([input_1, input_2])
            else:
                try:
                    input_ = Composition._postorder_eval(comp.children[0])
                except IndexError:
                    raise UnevaluableCompositionError(
                        "Not all required arrays are provided."
                    )  # from None

                return root.eval([input_])
        else:
            return root.eval()

    @staticmethod
    def _postorder_eval_with_indices(
        comp: "Composition",
    ) -> tuple[OutputWithIndicesType, int]:
        root: Function = comp._root_function
        if not root.is_evaluable():
            if root.definition is zip_with:
                try:
                    input_1, id_1 = Composition._postorder_eval_with_indices(
                        comp.children[0]
                    )
                except IndexError:
                    raise UnevaluableCompositionError(
                        "Not all required arrays are provided."
                    )  # from None

                if root.inputs:
                    return root.eval_with_indices([input_1], id_1=id_1)
                else:
                    try:
                        input_2, id_2 = Composition._postorder_eval_with_indices(
                            comp.children[1]
                        )
                    except IndexError:
                        raise UnevaluableCompositionError(
                            "Not all required arrays are provided."
                        )  # from None

                    return root.eval_with_indices(
                        [input_1, input_2], id_1=id_1, id_2=id_2
                    )
            else:
                try:
                    input_, id_ = Composition._postorder_eval_with_indices(
                        comp.children[0]
                    )
                except IndexError:
                    raise UnevaluableCompositionError(
                        "Not all required arrays are provided."
                    )  # from None

                return root.eval_with_indices([input_], id_1=id_)
        else:
            return root.eval_with_indices()

    @staticmethod
    def _preorder_leaves(comp: "Composition", leaves: list["Composition"]) -> None:
        if (
            not comp.children
            or comp._root_function.definition is zip_with
            and len(comp.children) == 1
        ):
            leaves.append(comp)

        for child in comp.children:
            Composition._preorder_leaves(child, leaves)

    @staticmethod
    def _preorder_num_inputs(comp: "Composition") -> int:
        root: Function = comp._root_function
        children: list["Composition"] = comp.children
        if root.definition is zip_with and not children:
            return 2
        elif root.definition is not zip_with and not children:
            return 1
        elif root.definition is zip_with and len(children) == 1:
            return 1 + Composition._preorder_num_inputs(children[0])
        else:
            return sum(Composition._preorder_num_inputs(child) for child in children)

    @staticmethod
    def _preorder_subcompositions(
        comp: "Composition", length_, subcomps: list["Composition"]
    ) -> None:
        curr_length = Composition._postorder_len(comp)
        if curr_length == length_:
            subcomps.append(comp)
        elif curr_length > length_:
            for child in comp.children:
                Composition._preorder_subcompositions(child, length_, subcomps)

    @staticmethod
    def _preorder_first_map(
        comp: "Composition", first_map: "Composition" = None
    ) -> Optional["Composition"]:
        if comp._root_function.definition is map_func:
            if comp.children:
                return Composition._preorder_first_map(comp.children[0], comp)
            else:
                return comp
        else:
            return first_map

    @staticmethod
    def _postorder_string(comp: "Composition") -> str:
        if not comp.children:
            return str(comp._root_function).replace("input", "final")
        else:
            string = str(comp._root_function)
            for child in comp.children:
                child_string = Composition._postorder_string(child)
                string = string.replace("input", child_string, 1)

            return string

    @staticmethod
    def _postorder_len(comp: "Composition") -> int:
        length_ = 1
        for child in comp.children:
            length_ += Composition._postorder_len(child)

        return length_

    # Staticmethods used for generating purposes
    @staticmethod
    def _postorder_sample_generation(
        comp: "Composition",
        state_tuple: tuple[Input, ...],
        output: OutputType,  # Output of the entire composition
        samples: list[dict],
        copy_ids: list[int],
        left_index_dict: dict[int, int],
    ) -> tuple[tuple[Input, ...], int]:
        if comp.children:
            ret_val = Composition._postorder_sample_generation(
                comp.children[0],
                state_tuple,
                output,
                samples,
                copy_ids,
                left_index_dict,
            )
            prev_state_tuple, index_1 = ret_val

            if index_1 is None:  # Copy encountered
                return Composition._handle_copied_subcomposition(
                    comp, prev_state_tuple, output, samples, copy_ids, left_index_dict
                )

            if len(comp.children) == 2:
                left_index_dict[id(comp)] = index_1
                ret_val = Composition._postorder_sample_generation(
                    comp.children[1],
                    prev_state_tuple,
                    output,
                    samples,
                    copy_ids,
                    left_index_dict,
                )
                index_1 = left_index_dict[id(comp)]
                del left_index_dict[id(comp)]  # Entry is no longer needed
                prev_state_tuple, index_2 = ret_val
                prev_result_1 = prev_state_tuple[index_1]

                try:
                    prev_result_2 = prev_state_tuple[index_2]
                    result = comp._root_function.eval(
                        [prev_result_1.data, prev_result_2.data]
                    )
                    result = Input(result)
                    result.id = comp._id
                except TypeError:  # None as index
                    return Composition._handle_copied_subcomposition(
                        comp,
                        prev_state_tuple,
                        output,
                        samples,
                        copy_ids,
                        left_index_dict,
                    )

                indices = sorted([index_1, index_2])
                new_state_tuple = Composition._get_new_state_tuple(
                    prev_state_tuple, result, indices
                )

                sample = Composition._get_sample(
                    prev_state_tuple, output, indices, comp._root_function.as_dict()
                )
                samples.append(sample)

                for key in left_index_dict:
                    if indices[1] < left_index_dict[key]:
                        left_index_dict[key] -= 1

                if comp._id in copy_ids:
                    new_state_tuple = Composition._add_copy_sample(
                        comp, new_state_tuple, output, samples, copy_ids, indices[0],
                    )

                return new_state_tuple, indices[0]
            elif comp._root_function.definition is zip_with:
                # Half-leaf zip_with node
                try:
                    prev_result = prev_state_tuple[index_1]
                except TypeError:
                    return Composition._handle_copied_subcomposition(
                        comp,
                        prev_state_tuple,
                        output,
                        samples,
                        copy_ids,
                        left_index_dict,
                    )

                result = comp._root_function.eval([prev_result.data])
                result = Input(result)
                result.id = comp._id

                input_2 = comp._root_function.inputs[0]
                index_2 = prev_state_tuple.index(input_2)
                indices = sorted([index_1, index_2])

                if input_2.id in copy_ids:
                    prev_state_tuple = Composition._add_copy_sample(
                        input_2, prev_state_tuple, output, samples, copy_ids, index_2
                    )

                new_state_tuple = Composition._get_new_state_tuple(
                    prev_state_tuple, result, indices
                )

                sample = Composition._get_sample(
                    prev_state_tuple, output, indices, comp._root_function.as_dict()
                )
                samples.append(sample)

                for key in left_index_dict:
                    if indices[1] < left_index_dict[key]:
                        left_index_dict[key] -= 1

                if comp._id in copy_ids:
                    new_state_tuple = Composition._add_copy_sample(
                        comp, new_state_tuple, output, samples, copy_ids, indices[0],
                    )

                return new_state_tuple, indices[0]
            else:
                try:
                    prev_result = prev_state_tuple[index_1]
                    result = comp._root_function.eval([prev_result.data])
                    result = Input(result)
                    result.id = comp._id
                except TypeError:  # None as index
                    return Composition._handle_copied_subcomposition(
                        comp,
                        prev_state_tuple,
                        output,
                        samples,
                        copy_ids,
                        left_index_dict,
                    )

                indices = [index_1]

                new_state_tuple = Composition._get_new_state_tuple(
                    prev_state_tuple, result, indices
                )

                if prev_state_tuple != new_state_tuple:
                    sample = Composition._get_sample(
                        prev_state_tuple, output, indices, comp._root_function.as_dict()
                    )
                    samples.append(sample)

                if comp._id in copy_ids:
                    new_state_tuple = Composition._add_copy_sample(
                        comp, new_state_tuple, output, samples, copy_ids, indices[0],
                    )

                return new_state_tuple, indices[0]
        else:
            return Composition._handle_leaf(
                comp, state_tuple, output, samples, copy_ids, left_index_dict
            )

    @staticmethod
    def _handle_leaf(
        comp: "Composition",
        state_tuple: tuple[Input, ...],
        output: OutputType,
        samples: list[dict],
        copy_ids: list[int],
        left_index_dict: dict[int, int],
    ) -> tuple[tuple[Input, ...], int]:
        root_function = comp._root_function

        try:
            if root_function.definition is zip_with:
                input_1, input_2 = root_function.inputs
                index_1 = state_tuple.index(input_1)
                index_2 = state_tuple.index(input_2)
                if index_1 == index_2:  # Both inputs are equivalent copies
                    # Another copy is needed
                    state_tuple = Composition._add_copy_sample(
                        input_1, state_tuple, output, samples, copy_ids, index_1
                    )
                    index_2 = state_tuple.index(input_2, index_1 + 1)
                indices = sorted([index_1, index_2])
            else:
                input_ = root_function.inputs[0]
                index = state_tuple.index(input_)
                indices = [index]
        except ValueError:  # Copy encountered
            return Composition._handle_copied_subcomposition(
                comp, state_tuple, output, samples, copy_ids, left_index_dict
            )

        for input_ in root_function.inputs:
            if input_.id in copy_ids:
                copy_ids.remove(input_.id)
                copy_index = state_tuple.index(input_)
                function_dict = {
                    "definition": "copy_state_tuple",
                    "num_lambda_number": None,
                    "num_lambda_operator": None,
                    "bool_lambda_number": None,
                    "bool_lambda_operator": None,
                    "take_drop_number": None,
                }
                sample = Composition._get_sample(
                    state_tuple, output, [copy_index], function_dict
                )
                samples.append(sample)
                state_tuple = copy_state_tuple(state_tuple, copy_index)

        result = Input(comp.eval())
        result.id = comp._id

        new_state_tuple = Composition._get_new_state_tuple(state_tuple, result, indices)

        for key in left_index_dict:
            if len(indices) == 2 and indices[1] < left_index_dict[key]:
                left_index_dict[key] -= 1

        if state_tuple != new_state_tuple:
            sample = Composition._get_sample(
                state_tuple, output, indices, comp._root_function.as_dict()
            )
            samples.append(sample)

        if comp._id in copy_ids:
            new_state_tuple = Composition._add_copy_sample(
                comp, new_state_tuple, output, samples, copy_ids, indices[0]
            )

        return new_state_tuple, indices[0]

    @staticmethod
    def _handle_copied_subcomposition(
        comp: "Composition",
        state_tuple: tuple[Input, ...],
        output: OutputType,
        samples: list[dict],
        copy_ids: list[int],
        left_index_dict: dict[int, int],
    ) -> tuple[tuple[Input, ...], int]:
        result = comp.eval()
        result = Input(result)
        result.id = comp._id
        if result in state_tuple:
            ret_index = state_tuple.index(result)
            for key in left_index_dict:
                if left_index_dict[key] == ret_index:
                    ret_index = state_tuple.index(result, ret_index + 1)
                    break
            if comp._id in copy_ids:
                state_tuple = Composition._add_copy_sample(
                    comp, state_tuple, output, samples, copy_ids, ret_index
                )
        else:
            ret_index = None

        return state_tuple, ret_index

    @staticmethod
    def _add_copy_sample(
        comp: Union["Composition", Input],
        new_state_tuple: tuple[Input, ...],
        output: OutputType,
        samples: list[dict],
        copy_ids: list[int],
        index: int,
    ) -> tuple[Input, ...]:
        copy_ids.remove(comp.id)
        function_dict = {
            "definition": "copy_state_tuple",
            "num_lambda_number": None,
            "num_lambda_operator": None,
            "bool_lambda_number": None,
            "bool_lambda_operator": None,
            "take_drop_number": None,
        }

        sample = Composition._get_sample(
            new_state_tuple, output, [index], function_dict
        )
        samples.append(sample)
        return copy_state_tuple(new_state_tuple, index)

    @staticmethod
    def _get_new_state_tuple(
        state_tuple: tuple[Input, ...],
        new_element: Union[Input, InputType],
        indices: list[int],
    ) -> tuple[Input, ...]:
        if len(indices) == 1:
            new_state_tuple = (
                state_tuple[: indices[0]]
                + (new_element,)
                + state_tuple[indices[0] + 1 :]
            )
        else:
            new_state_tuple = (
                state_tuple[: indices[0]]
                + (new_element,)
                + state_tuple[indices[0] + 1 : indices[1]]
                + state_tuple[indices[1] + 1 :]
            )

        return new_state_tuple

    @staticmethod
    def _get_sample(
        input_: tuple[Input, ...],
        output: OutputType,
        indices: list[int],
        function_dict: dict,
    ) -> dict:
        raw_input = tuple([input_elem.data for input_elem in input_])

        optimized_indices = []
        for index in indices:
            elem = raw_input[index]
            index_candidate = raw_input.index(elem)
            if index_candidate in optimized_indices:
                index_candidate = raw_input.index(elem, index_candidate + 1)
            optimized_indices.append(index_candidate)

        return {
            "input": raw_input,
            "output": (output,),
            "next_transformation": {
                "indices": optimized_indices,
                "function": function_dict,
            },
        }

    @staticmethod
    def _optimize_drop_filter(comp: "Composition") -> "Composition":
        for ind, child in enumerate(comp.children):
            comp.children[ind] = Composition._optimize_drop_filter(comp=child)

        try:
            result = comp.eval()
        except ValueError:
            if (
                comp._root_function.definition in {min, max}
                and comp.parent is None
                and comp.children
            ):
                comp.children[0]._parent = None
                return comp.children[0]
            raise
        is_empty_result = not isinstance(result, int) and all(x == [] for x in result)

        if comp._root_function.definition is drop and is_empty_result:
            comp = Composition._improve_drop(comp)  # Reference might change
        elif comp._root_function.definition is take:
            Composition._improve_take(comp)
        elif comp._root_function.definition is filter_func:
            Composition._improve_filter(comp)

        return comp

    @staticmethod
    def _improve_filter(comp: "Composition") -> None:
        function = comp._root_function
        function_input = comp._get_input()

        function_result = comp.eval()
        is_empty_result = all(x == [] for x in function_result)
        is_same_result = function_input == function_result
        # We consider is_same_result an opportunity for optimization,
        # because we can tweak the parameters of filter in order to
        # receive a different output than the input.
        # E.g. in the case of reverse, we couldn't optimize anything
        # as it doesn't have any parameters. Thus, a reverse function
        # that doesn't change its input is simply discarded in the
        # as_samples method.

        if is_empty_result or is_same_result:
            Composition._handle_empty_or_noop_filter(comp)
            function_result = comp.eval()

        if function.operator is operator.lt:
            for num_candidate in range(function.number - 1, grammar.MIN_NUM - 1, -1):
                function.number = num_candidate
                if comp.eval() != function_result:
                    function.number += 1
                    break
        elif function.operator is operator.gt:
            for num_candidate in range(function.number + 1, grammar.MAX_NUM + 1):
                function.number = num_candidate
                if comp.eval() != function_result:
                    function.number -= 1
                    break

    @staticmethod
    def _improve_take(comp: "Composition") -> None:
        function = comp._root_function
        function_input = comp._get_input()

        original_output = comp.eval()
        if original_output != function_input:
            return

        for num_candidate in range(function.number - 1, 0, -1):
            function.number = num_candidate
            output = comp.eval()
            if output != original_output:
                break

    @staticmethod
    def _improve_drop(comp: "Composition") -> "Composition":
        function = comp._root_function
        for num_candidate in range(function.number - 1, 0, -1):
            function.number = num_candidate
            result = comp.eval()
            if result and any(x != [] for x in result):
                return comp

        # Node cannot be saved with current inputs
        return Composition._delete_drop_node(comp)

    @staticmethod
    def _delete_drop_node(comp: "Composition") -> "Composition":
        if comp.parent:
            index = comp._child_index(comp.parent, comp)
            if comp.children:
                comp.parent.children[index] = comp.children[0]
            else:
                del comp.parent.children[index]
                comp.parent._root_function.inputs.insert(
                    0, comp._root_function.inputs[0]
                )
            return comp.children[0]
        else:
            if not comp.children:
                raise UncorrectableSampleError(
                    "Composition with provided input always returns an empty output and "
                    "no function can be deleted, as the result would be an empty "
                    "composition."
                )
            comp.children[0]._parent = None
            return comp.children[0]

    @staticmethod
    def _handle_empty_or_noop_filter(comp: "Composition") -> None:
        function = comp._root_function
        if function.operator in {operator.gt, operator.lt}:
            success = Composition._try_operator_lt_gt(comp, function.operator)
            if not success:
                success = Composition._try_operator_mod(comp)
                if not success:
                    success = Composition._try_operator_eq(comp)
                    if not success:
                        Composition._try_operator_lt_gt_noop(comp)
        elif function.operator is operator.eq:
            success = Composition._try_operator_eq(comp)
            if not success:
                success = Composition._try_operator_lt_gt(comp, operator.gt)
                if not success:
                    success = Composition._try_operator_mod(comp)
                    if not success:
                        Composition._try_operator_lt_gt_noop(comp)
        else:
            success = Composition._try_operator_mod(comp)
            if not success:
                success = Composition._try_operator_eq(comp)
                if not success:
                    success = Composition._try_operator_lt_gt(comp, operator.lt)
                    if not success:
                        Composition._try_operator_lt_gt_noop(comp)

    @staticmethod
    def _try_operator_lt_gt_noop(comp: "Composition"):
        function = comp._root_function
        function.operator = operator.gt

        if all(x == [] for x in comp.eval()):
            function.operator = operator.lt

    # noinspection PyTypeChecker
    @staticmethod
    def _try_operator_lt_gt(comp: "Composition", op: Callable) -> bool:
        function = comp._root_function
        function_input = comp._get_input()

        if isinstance(function_input[0], int):
            # 1 I/O
            min_elem = min(function_input)
            max_elem = max(function_input)
        else:
            # Clear out empty inputs before mapping min
            # noinspection PyTypeChecker
            min_elem = max(map(min, filter(lambda x: x, function_input)))
            max_elem = min(map(max, filter(lambda x: x, function_input)))

        if op is operator.lt:
            lob = max(min_elem + 1, grammar.MIN_NUM)
            hib = min(max_elem, grammar.MAX_NUM)
        else:
            lob = max(min_elem, grammar.MIN_NUM)
            hib = min(max_elem - 1, grammar.MAX_NUM)
        possible = hib - lob >= 0  # Non-empty interval

        if possible:
            function.operator = op
            function.number = hib

        return possible

    @staticmethod
    def _try_operator_mod(comp: "Composition") -> bool:
        function = comp._root_function
        function_input = comp._get_input()

        if isinstance(function_input[0], int):
            for i in range(2, grammar.MAX_NUM + 1):
                div_list = []
                for num in function_input:
                    # noinspection PyTypeChecker
                    if num % i == 0:
                        div_list.append(num)
                if div_list != [] and div_list != function_input:
                    function.operator = operator.mod
                    function.number = i
                    return True
            else:
                return False
        else:
            set_list = []
            for sample in function_input:
                sample_set = set()
                for i in range(2, grammar.MAX_NUM + 1):
                    div_list = []
                    for num in sample:
                        if num % i == 0:
                            div_list.append(num)
                    if div_list != [] and div_list != sample:
                        sample_set = sample_set | {i}
                set_list.append(sample_set)
            result_candidates = reduce(lambda x, y: x & y, set_list)
            if result_candidates:
                function.operator = operator.mod
                function.number = min(result_candidates)
            return bool(result_candidates)

    @staticmethod
    def _try_operator_eq(comp: "Composition") -> bool:
        function = comp._root_function
        function_input = comp._get_input()

        if isinstance(function_input[0], int):
            result_candidates = [
                elem
                for elem in function_input
                if elem in range(grammar.MIN_NUM, grammar.MAX_NUM + 1)
            ]
            if result_candidates:
                function.operator = operator.eq
                new_num = Counter(result_candidates).most_common(1)[0][0]
                function.number = new_num
            return bool(result_candidates)
        else:
            set_list = []
            for sample in function_input:
                unique_numbers = set(sample)
                filtered_unique_numbers = {
                    num
                    for num in unique_numbers
                    if num in range(grammar.MIN_NUM, grammar.MAX_NUM + 1)
                }
                set_list.append(filtered_unique_numbers)
            result_candidates = reduce(lambda x, y: x & y, set_list)

            if result_candidates:
                function.operator = operator.eq

                occurrences: list[list[int]] = []
                for candidate in result_candidates:
                    occurrences.append([])
                    for sample in function_input:
                        occurrences[-1].append(sample.count(candidate))

                tuple_list = list(zip(result_candidates, occurrences))
                mapped_tuple_list = map(
                    lambda x: (x[0], min(x[1]), sum(x[1])), tuple_list
                )

                function.number = max(mapped_tuple_list, key=lambda x: x[1:])[0]
            return bool(result_candidates)

    @staticmethod
    def _child_index(
        parent: "Composition", child: "Composition", start_index: int = 0
    ) -> int:
        return [i for i, x in enumerate(parent.children) if x is child][start_index]

    def _check_node_output_in_range(self, out_range: Iterable) -> Optional[OutputType]:
        result = self.eval()

        if isinstance(result, int):
            if result in out_range:
                return None
            return result
        elif isinstance(result, list):
            if result and isinstance(result[0], int):
                if all(r in out_range for r in result):
                    return None
                else:
                    return result
            elif result:  # list[list[int]]
                if all(all(r in out_range for r in res) for res in result):
                    return None
                else:
                    return result

            with open("fucked.pkl", "wb") as f:
                import dill

                dill.dump(self, f)

            raise UnreachableCodeError
        else:
            raise UnreachableCodeError

    @staticmethod
    def _fix_input(problem_node: "Composition"):
        def find_input_by_id(node: "Composition", input_id: int) -> InputType:
            for sub_node in node:
                for inp in sub_node._root_function.inputs:
                    if inp.id == input_id:
                        return inp.data

        def weight(output_list: list[int]) -> int:
            ret = 0
            for elem in output_list:
                if elem > config.OUTPUT_HIB:
                    ret += elem - config.OUTPUT_HIB
                elif elem < config.OUTPUT_LOB:
                    ret += config.OUTPUT_LOB - elem

            return ret

        def extent_of_protrusion(node: Composition, ind=None) -> int:
            temp = node.eval()
            if isinstance(temp, int):
                return weight([temp])
            elif ind is not None:
                temp_temp = temp[ind]
                if isinstance(temp_temp, int):
                    return weight([temp_temp])
                return weight(temp_temp)
            else:
                return weight(temp)

        def handle_small_protrusion_helper(
            out,
            problem_node_,
            problem_input,
            magic_number,
            input_id,
            problem_index,
            is_more_io,
        ):
            iterator = out[problem_index] if is_more_io else out

            for sub_out, ind in iterator:
                if not (config.OUTPUT_LOB <= sub_out <= config.OUTPUT_HIB):
                    handle_protrusion(
                        problem_node_,
                        problem_index,
                        problem_input,
                        is_more_io,
                        ind,
                        magic_number,
                        input_id,
                    )

        def handle_protrusion(
            problem_node_,
            problem_index,
            problem_input,
            is_more_io,
            ind,
            magic_number,
            input_id,
        ):
            base_line = extent_of_protrusion(problem_node_, problem_index)

            first_try = problem_input.copy()
            if is_more_io:
                first_try[problem_index][ind] += magic_number
            else:
                first_try[ind] += magic_number

            problem_node_.fill_input_by_id(first_try, input_id)

            eval_1_protrusion = extent_of_protrusion(problem_node_, problem_index)

            if eval_1_protrusion < base_line:
                if is_more_io:
                    first_try[problem_index][ind] += magic_number
                else:
                    first_try[ind] += magic_number
            else:
                if is_more_io:
                    first_try[problem_index][ind] -= magic_number
                else:
                    first_try[ind] -= magic_number

        def handle_small_protrusion(
            out, problem_node_, input_id, problem_input, magic_number
        ):
            is_more_io = Input.is_more_io_data(problem_input)
            if is_more_io:
                for i, _ in enumerate(problem_input):
                    handle_small_protrusion_helper(
                        out,
                        problem_node_,
                        problem_input,
                        magic_number,
                        input_id,
                        i,
                        is_more_io,
                    )
            else:
                handle_small_protrusion_helper(
                    out,
                    problem_node_,
                    problem_input,
                    magic_number,
                    input_id,
                    None,
                    is_more_io,
                )

            return problem_input

        def numeric_protrusion_helper(
            problem_input, problem_node_, is_more_io, id_, magic_number, problem_index,
        ):
            for ind in range(len(problem_input)):
                handle_protrusion(
                    problem_node_,
                    problem_index,
                    problem_input,
                    is_more_io,
                    ind,
                    magic_number,
                    id_,
                )

        def handle_protrusion_in_numeric_func(
            problem_node_, id_, problem_input, magic_number
        ):

            is_more_io = Input.is_more_io_data(problem_input)

            if is_more_io:
                for i, _ in enumerate(problem_input):
                    numeric_protrusion_helper(
                        problem_input, problem_node_, is_more_io, id_, magic_number, i,
                    )
            else:
                numeric_protrusion_helper(
                    problem_input, problem_node_, is_more_io, id_, magic_number, None,
                )

            return problem_input

        def h_helper(out, problem_input, problem_index, is_more_io):
            iterator = out[problem_index] if is_more_io else out
            for sub_out_, ind_ in iterator:
                if not (config.OUTPUT_LOB <= sub_out_ <= config.OUTPUT_HIB):
                    if is_more_io:
                        problem_input[problem_index][ind_] = utils.symmetric_floordiv(
                            problem_input[problem_index][ind_], 2
                        )
                    else:
                        problem_input[ind_] = utils.symmetric_floordiv(
                            problem_input[ind_], 2
                        )

        def handle_zipwith_mul_protrusion(out, problem_input):
            is_more_io = Input.is_more_io_data(problem_input)
            if is_more_io:
                for i, _ in enumerate(problem_input):
                    h_helper(out, problem_input, i, is_more_io)
            else:
                h_helper(out, problem_input, None, is_more_io)

            return problem_input

        magic_number_ = 10

        out_, input_id_ = problem_node.eval_with_indices()
        problem_input_ = find_input_by_id(problem_node, input_id_)
        root_func: Function = problem_node._root_function

        if root_func.definition is zip_with:
            if root_func.operator in (operator.sub, operator.add):

                problem_input_ = handle_small_protrusion(
                    out_, problem_node, input_id_, problem_input_, magic_number_,
                )
                problem_node.fill_input_by_id(problem_input_, input_id_)

            elif root_func.operator is operator.mul:

                problem_input_ = handle_zipwith_mul_protrusion(out_, problem_input_)
                problem_node.fill_input_by_id(problem_input_, input_id_)
        elif root_func.definition is map_func:
            problem_input_ = handle_small_protrusion(
                out_, problem_node, input_id_, problem_input_, magic_number_
            )
            problem_node.fill_input_by_id(problem_input_, input_id_)
        elif root_func.definition in (sum, max, min, length):
            problem_input_ = handle_protrusion_in_numeric_func(
                problem_node, input_id_, problem_input_, magic_number_
            )
            problem_node.fill_input_by_id(problem_input_, input_id_)

    @staticmethod
    def _bottom_up_random_inputs(comp: "Composition") -> tuple[bool, "Composition"]:
        out_range = range(config.OUTPUT_LOB, config.OUTPUT_HIB + 1)
        problem_node = None

        for node in comp.postorder_iterate():
            problem_output = node._check_node_output_in_range(out_range)
            if problem_output is not None:
                problem_node = node
                break

        if problem_node is None:
            return True, comp

        Composition._fix_input(problem_node)

        comp = Composition._optimize_drop_filter(comp)

        return False, comp

    def _naive_fill_inputs(self, input_dict: dict, num_io_examples: int):
        for input_ in self.leaf_inputs:
            if input_.id not in input_dict:
                inputs = []
                for _ in range(num_io_examples):
                    list_length = random.randint(
                        config.INPUT_LENGTH_LOB, config.INPUT_LENGTH_HIB
                    )
                    new_input_list = np.random.randint(
                        low=config.INPUT_LOB,
                        high=config.INPUT_HIB + 1,
                        size=list_length,
                    ).tolist()
                    inputs.append(new_input_list)

                if num_io_examples == 1:
                    inputs = inputs[0]

                input_dict[input_.id] = inputs
                input_.data = inputs
            else:
                input_.data = input_dict[input_.id]

    @staticmethod
    def _postorder_copy_ids(comp: "Composition", ret_dict: dict[int, int]) -> None:
        for child in comp.children:
            Composition._postorder_copy_ids(child, ret_dict)

        if comp.parent is None or ret_dict[comp.parent._id] == 0:
            ret_dict[comp._id] += 1

            if ret_dict[comp._id] == 1:
                for input_ in comp._root_function.inputs:
                    ret_dict[input_.id] += 1


def main():
    print("# 1")
    take_2 = Function.from_dict({"function": "take", "number": "2"})
    take_2.inputs = [Input()]
    map_ = Function.from_dict({"function": "map_func", "operator": "*", "number": "2"})
    take_3 = Function.from_dict({"function": "take", "number": "3"})
    take_3.inputs = [Input()]
    zw_minus = Function.from_dict({"function": "zip_with", "operator": "-"})
    zw_mul = Function.from_dict({"function": "zip_with", "operator": "*"})

    take_2 = Composition(take_2)
    map_ = Composition.from_composition(map_, take_2)
    map_plus1 = Composition.from_composition(
        Function.from_dict({"function": "map_func", "operator": "+", "number": "1"}),
        map_,
    )

    take_3 = Composition(take_3)
    zw_minus = Composition.from_composition(zw_minus, map_plus1, take_3)

    zw_mul = Composition.from_composition(zw_mul, map_, zw_minus)
    print(zw_mul)

    utils.visualize(zw_mul, "1.png")

    zw_mul = Composition.fill_random_inputs(zw_mul, 1)

    for sample_ in zw_mul.as_samples():
        print(sample_)

    print("# 2")
    take_2 = Composition(Function.from_dict({"function": "take", "number": "2"}))
    take_2.root_function.inputs = [Input([1, 2, 3, 4])]
    zw_mul = Composition.from_composition(
        Function.from_dict({"function": "zip_with", "operator": "*"}), take_2, take_2
    )
    comp__ = Composition.from_composition(
        Function.from_dict({"function": "zip_with", "operator": "+"}), zw_mul, take_2
    )

    print(comp__)

    for sample_ in comp__.as_samples():
        print(sample_)

    utils.visualize(comp__, "2.png")

    print("# 3")
    inp = Input([1, 2, 3, 4])
    take__ = Composition(Function.from_dict({"function": "take", "number": "2"}))
    take__.root_function.inputs = [inp]
    drop__ = Composition(Function.from_dict({"function": "drop", "number": "2"}))
    drop__.root_function.inputs = [inp]

    comp__ = Composition.from_composition(
        Function.from_dict({"function": "zip_with", "operator": "*"}), take__, drop__
    )

    print(comp__)

    for sample_ in comp__.as_samples():
        print(sample_)

    utils.visualize(comp__, "3.png")

    print("# 4")
    inp = Input([1, 2, 3, 4])
    take__ = Composition(Function.from_dict({"function": "take", "number": "2"}))
    take__.root_function.inputs = [inp]

    zwplus = Composition(Function.from_dict({"function": "zip_with", "operator": "+"}))
    zwplus.root_function.inputs = [inp, Input([1, 2, 3, 4])]

    zwmul = Composition.from_composition(
        Function.from_dict({"function": "zip_with", "operator": "*"}), take__, zwplus
    )

    print(zwmul)

    for sample_ in zwmul.as_samples():
        print(sample_)

    utils.visualize(zwmul, "4.png")

    print("# 5")
    inp = Input([1, 2, 3, 4])
    take__ = Composition(Function.from_dict({"function": "take", "number": "2"}))
    take__.root_function.inputs = [inp]

    zwplus = Composition(Function.from_dict({"function": "zip_with", "operator": "+"}))
    zwplus.root_function.inputs = [inp, inp]

    zwmul = Composition.from_composition(
        Function.from_dict({"function": "zip_with", "operator": "*"}), take__, zwplus
    )

    print(zwmul)

    for sample_ in zwmul.as_samples():
        print(sample_)

    utils.visualize(zwmul, "5.png")

    print("# 6")
    inp = Input([[1, 2, 3, 4], [1, 2, 3, 4]])
    inp_2 = Input([[69, 69, 69] * 3, [69, 69, 69] * 3])

    zwplus = Composition(Function.from_dict({"function": "drop", "number": "2"}))
    zwplus.root_function.inputs = [inp]
    zwplus = Composition.from_composition(
        Function.from_dict({"function": "reverse_func"}), zwplus
    )

    zwplus_2 = Composition(
        Function.from_dict({"function": "map_func", "operator": "*", "number": "2"})
    )
    zwplus_2.root_function.inputs = [inp_2]

    zwmul = Composition.from_composition(
        Function.from_dict({"function": "zip_with", "operator": "*"}), zwplus, zwplus_2
    )

    print(zwmul)

    for sample_ in zwmul.as_samples():
        print(sample_)

    print(zwmul.eval_with_indices())

    utils.visualize(zwmul, "6.png")

    print("# 7")
    map_copy = Composition(
        Function.from_dict({"function": "map_func", "operator": "*", "number": "2"})
    )
    map_copy.root_function.inputs.append(Input([1337]))
    zwminus = Composition.from_composition(
        Function.from_dict({"function": "zip_with", "operator": "-"}), map_copy
    )
    zwminus.root_function.inputs.append(Input([1337]))

    comp_ = Composition.from_composition(
        Function.from_dict({"function": "zip_with", "operator": "*"}), map_copy, zwminus
    )

    print(comp_)

    for sample_ in comp_.as_samples():
        print(sample_)

    utils.visualize(comp_, "7.png")

    print("#8")
    inp_1 = Input([1])
    zw_1 = Composition(Function.from_dict({"function": "zip_with", "operator": "min"}))
    zw_1.root_function.inputs = [inp_1, inp_1]
    filter__ = Composition.from_composition(
        Function.from_dict(
            {"function": "filter_func", "operator": ">", "number": "-8"}
        ),
        zw_1,
    )
    map_copy = Composition(
        Function.from_dict({"function": "map_func", "operator": "*", "number": "-8"})
    )
    map_copy.root_function.inputs = [Input([2])]
    zw_2 = Composition.from_composition(
        Function.from_dict({"function": "zip_with", "operator": "*"}),
        filter__,
        map_copy,
    )
    zwmul = Composition.from_composition(
        Function.from_dict({"function": "zip_with", "operator": "*"}), zw_2, map_copy
    )

    print(zwmul)
    print(zwmul.copy_ids)

    utils.visualize(zwmul, "8.png")

    for sample_ in zwmul.as_samples():
        print(sample_)

    map_mul_2 = Composition(
        Function.from_dict({"function": "map_func", "operator": "*", "number": "2"})
    )
    map_mul_2.root_function.inputs = [Input()]
    map_plus_1 = Composition(
        Function.from_dict({"function": "map_func", "operator": "+", "number": "1"})
    )
    map_plus_1.root_function.inputs = [Input()]
    comp = Composition.from_composition(
        Function.from_dict({"function": "zip_with", "operator": "*"}),
        map_mul_2,
        map_plus_1,
    )

    print(comp)
    comp = Composition.fill_random_inputs(comp, 1)
    print(comp)


if __name__ == "__main__":
    main()
