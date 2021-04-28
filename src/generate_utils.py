import copy
import itertools
import json
import math
import operator
import random
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations, permutations
from typing import Optional, Union, Generator
from src.grammar import ABBREVATION_DICT

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, Qt

# import src.utils
from src.composition import Composition, UnimprovableInputError
from src.func_impl import *
from src.function import Function, ALL_FUNCTIONS, BUCKETS


# python -m src.generate_utils 100 --functions 6 --inputs 4 --unique_inputs 2


@dataclass
class Arguments:
    number: int
    filename: str
    functions: int
    io: int
    inputs: int
    unique_inputs: int
    num_samples_per_comp: int
    is_test: bool


class NotReachable(Exception):
    pass


class UnsatisfiableParametersError(Exception):
    pass


@dataclass
class CompDict:
    args: Arguments
    _dict: defaultdict[
        int, defaultdict[int, defaultdict[bool, list[Composition]]]
    ] = field(
        default_factory=lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
    )
    finalized: bool = False

    def __iter__(self):
        return iter(self._dict)

    def __post_init__(self):
        one_length = defaultdict(
            lambda: defaultdict(list)
        )  # Keys: number of unique inputs
        for function in ALL_FUNCTIONS:
            function = copy.deepcopy(function)
            if self.args.inputs > 1 and function.definition is zip_with:
                comp = Composition(function)
                comp.root_function.inputs = [Input(), Input()]
                one_length[2][False].append(comp)
            elif function.definition is not zip_with:
                comp = Composition(function)
                comp.root_function.inputs = [Input()]
                if function.definition in {sum, max, min, length}:
                    at_end = True
                else:
                    at_end = False
                one_length[1][at_end].append(comp)

        self._dict[1] = one_length

    def __getitem__(self, ind: Union[int, tuple[int, int, bool], tuple[int, int]]):
        if isinstance(ind, tuple) and len(ind) == 2:
            if self.finalized and (
                ind[0] not in self._dict.keys()
                or ind[1] not in self._dict[ind[0]].keys()
            ):
                raise IndexError("Invalid indices specified in a frozen CompDict:", ind)
            return self._dict[ind[0]][ind[1]][False] + self._dict[ind[0]][ind[1]][True]
        if isinstance(ind, tuple) and len(ind) == 3:
            if self.finalized and (
                ind[0] not in self._dict.keys()
                or ind[1] not in self._dict[ind[0]].keys()
                or ind[2] not in self._dict[ind[0]][ind[1]].keys()
            ):
                raise IndexError("Invalid indices specified in a frozen CompDict:", ind)
            return self._dict[ind[0]][ind[1]][ind[2]]
        elif isinstance(ind, int):
            if self.finalized and ind not in self._dict.keys():
                raise IndexError("Invalid index specified in a frozen CompDict:", ind)
            return self._dict[ind]
        else:
            raise IndexError

    def finalize(self):
        for key in list(self._dict.keys()):
            if not self._dict[key]:
                del self._dict[key]
            else:
                for key_2 in list(self._dict[key].keys()):
                    if not self._dict[key][key_2]:
                        del self._dict[key][key_2]
                    else:
                        for key_3 in list(self._dict[key][key_2].keys()):
                            if not self._dict[key][key_2][key_3]:
                                del self._dict[key][key_2][key_3]
        self.finalized = True

    def num_compositions_at_len(self, length_: int) -> int:
        if length_ not in self._dict.keys():
            return 0
        return sum(len(self[length_, key]) for key in self[length_].keys())

    def num_target_compositions(self) -> int:
        if (
            self.args.functions not in self._dict.keys()
            or self.args.inputs not in self._dict[self.args.functions].keys()
        ):
            return 0
        return len(self[self.args.functions, self.args.inputs])

    @property
    def target(self) -> list[Composition]:
        return self[self.args.functions, self.args.inputs][: self.args.number]

    def __str__(self):
        ret_str = "CompDict:\n"
        for key_1 in self._dict:
            ret_str += f"Length {key_1}:\n"
            for key_2 in self._dict[key_1]:
                ret_str += f"\tInputs {key_2}: {len(self[key_1, key_2])}\n"
                occurrence_dict = defaultdict(int)
                for comp in self[key_1, key_2]:
                    for node in comp:
                        occurrence_dict[node.root_function.definition] += 1
                for func_key in occurrence_dict:
                    ret_str += f"\t\t{func_key.__name__}: {occurrence_dict[func_key]}\n"

        return ret_str

    def statistics(self):
        occurrence_dict = defaultdict(int)
        for key_1 in self._dict:
            for key_2 in self._dict[key_1]:
                for comp in self[key_1, key_2]:
                    for node in comp:
                        func_name = node.root_function.definition.__name__
                        occurrence_dict[ABBREVATION_DICT[func_name]] += 1

        return occurrence_dict


@dataclass
class InputPlaceholder:
    copied_from: Optional["InputPlaceholder"] = None
    input: Optional[Input] = None
    occurrence: int = 1


@dataclass
class CompPlaceholder:
    inputs: list[InputPlaceholder]
    copied_from: Optional["CompPlaceholder"] = None
    composition: Optional[Composition] = None
    occurrence: int = 1


def sample_generator(args: Arguments):
    comp_dict = CompDict(args=args)
    generate_copyless_comps(comp_dict, args.number, args)
    comp_dict.finalize()

    with open("statistics.json", "w") as f:
        json.dump(comp_dict.statistics(), f)

    if args.inputs == args.unique_inputs:
        for comp in comp_dict.target:
            curr_comp_num_samples = 0
            deadlock_counter = 0
            while curr_comp_num_samples != args.num_samples_per_comp:
                try:
                    samples = comp.as_samples(args.io)
                    if args.is_test:
                        samples = samples[:1]
                        del samples[0]["next_transformation"]
                    yield samples
                    curr_comp_num_samples += 1
                except UnimprovableInputError:
                    deadlock_counter += 1

                if deadlock_counter == 100:
                    break
    else:
        yield from generate_copy_comps(comp_dict, args.number, args)


def generate_copyless_comps(comp_dict, number, args: Arguments):
    if (
        args.inputs <= 0
        or args.unique_inputs <= 0
        or args.unique_inputs > args.inputs
        or args.functions == 1
        and args.inputs > 2
    ):
        raise ValueError("Invalid number of inputs received.")

    if args.io < 1:
        raise ValueError("Invalid number of I/O examples received.")

    if args.functions == 1:
        return comp_dict

    one_input = args.inputs == args.unique_inputs == 1
    hib = args.functions + 1
    no_copy = args.inputs == args.unique_inputs

    for comp_length in range(2, hib):
        is_last_func = comp_length == args.functions
        expand_single_branch(
            comp_dict, comp_length, is_last_func, no_copy, number, args
        )

        if one_input:
            continue

        expand_multiple_branches(comp_dict, comp_length, is_last_func, number, args)

        if comp_dict.num_target_compositions() >= number:
            return comp_dict

    return comp_dict


def expand_single_branch(
    comp_dict: CompDict,
    comp_length: int,
    is_last_func: bool,
    no_copy: bool,
    number: int,
    args: Arguments,
):
    # Compositions of length `comp_length`, root_function is not zip_with
    if is_last_func and no_copy:
        for comp in comp_dict[comp_length - 1, args.inputs, False][:]:
            expand_single_branch_helper(
                comp_dict, comp, args, number - comp_dict.num_target_compositions(),
            )
    elif is_last_func:
        for comp in comp_dict[comp_length - 1, args.inputs, False][:]:
            expand_single_branch_helper(comp_dict, comp, args, config.VARIABILITY)
    else:
        for key in list(comp_dict[comp_length - 1].keys()):
            for comp in comp_dict[comp_length - 1, key, False][:]:
                expand_single_branch_helper(
                    comp_dict, comp, args, config.VARIABILITY,
                )


def expand_single_branch_helper(comp_dict, comp, args: Arguments, num_new_funcs):
    def handle_function(func):
        new_comp = Composition.from_composition(func, comp)
        at_end = func.definition in {sum, max, min, length}
        comp_dict[len(new_comp), new_comp.num_inputs, at_end].append(new_comp)

    if num_new_funcs <= 0:
        return
    elif num_new_funcs > len(ALL_FUNCTIONS):
        for new_func in ALL_FUNCTIONS:
            if new_func.definition is zip_with and args.inputs == comp.num_inputs:
                continue
            new_func = copy.deepcopy(new_func)
            if new_func.definition is zip_with:
                new_func.inputs = [Input()]
            handle_function(new_func)
    else:
        new_functions = []

        while len(new_functions) != num_new_funcs:
            if args.inputs > comp.num_inputs:
                new_func = choose_function()
                if new_func.definition is zip_with:
                    new_func.inputs = [Input()]
            else:
                new_func = choose_function(no_zip_with=True)

            if new_func not in new_functions:
                handle_function(new_func)
                new_functions.append(new_func)

        # Surely generate all possible length-input combinations
        if len(comp) == 1 and comp.root_function.definition is zip_with:
            for new_func in BUCKETS[zip_with]:
                if new_func not in new_functions:
                    new_func = copy.deepcopy(new_func)
                    new_func.inputs = [Input()]
                    handle_function(new_func)


def expand_multiple_branches(
    comp_dict: CompDict,
    comp_length: int,
    is_last_func: bool,
    number: int,
    args: Arguments,
):
    for level_1 in range(1, comp_length - 1):
        level_2 = comp_length - level_1 - 1  # Leave room for the zip_with

        prev_len = comp_dict.num_compositions_at_len(comp_length)
        curr_len = prev_len
        deadlock_counter = 0
        target_curr_len = 0

        while curr_len - prev_len < config.VARIABILITY ** (comp_length / 2):
            if deadlock_counter > 100:
                raise UnsatisfiableParametersError(
                    "Required parameters are unlikely to be satisfied."
                )

            for i in comp_dict[level_1].keys():
                for j in comp_dict[level_2].keys():
                    if (
                        not is_last_func
                        and i + j <= args.inputs
                        or is_last_func
                        and i + j == args.inputs
                    ):
                        comp_1 = random.choice(comp_dict[level_1, i, False])
                        comp_2 = random.choice(
                            comp_dict[level_2, j, False]
                        ).copy_with_new_ids()

                        new_func = choose_function(surely_zip_with=True)

                        new_comp = Composition.from_composition(
                            new_func, comp_1, comp_2
                        )

                        len_new_comp = len(new_comp)
                        new_comp_num_inputs = new_comp.num_inputs
                        if (
                            args.functions - len_new_comp
                            >= args.inputs - new_comp_num_inputs
                        ):
                            comp_dict[len_new_comp, new_comp_num_inputs, False].append(
                                new_comp
                            )

            if is_last_func:
                if target_curr_len == comp_dict.num_target_compositions():
                    deadlock_counter += 1
                else:
                    deadlock_counter = 0
            else:
                if curr_len == comp_dict.num_compositions_at_len(comp_length):
                    deadlock_counter += 1
                else:
                    deadlock_counter = 0

            curr_len = comp_dict.num_compositions_at_len(comp_length)
            target_curr_len = comp_dict.num_target_compositions()

            if comp_dict.num_target_compositions() >= number:
                return


def choose_function(
    surely_zip_with: bool = False, no_zip_with: bool = False
) -> Function:
    """Returns a random function from all the possible Function instances
    with the function types uniformly distributed.
    """
    if surely_zip_with and no_zip_with:
        raise ValueError("Surely_zip_with and no_zip_with contradict each other.")

    if not surely_zip_with and not no_zip_with:
        bucket = random.choice(list(BUCKETS.keys()))
    elif surely_zip_with:  # no_zip_with is false
        bucket = zip_with
    else:  # not surely_zip_with and no_zip_with
        bucket = random.choice(list(BUCKETS.keys() - {zip_with}))

    return copy.deepcopy(random.choice(BUCKETS[bucket]))


def check_num_of_copies(comb: list[CompPlaceholder], max_copy) -> bool:
    unfinished_copy_counter = 0
    for prev_comp in comb:
        if prev_comp.copied_from:
            unfinished_copy_counter += 1
        else:
            for i in prev_comp.inputs:
                if i.copied_from:
                    unfinished_copy_counter += 1

    return unfinished_copy_counter <= max_copy


def uniques(placeholder_list: list[CompPlaceholder]):
    counter = 0
    for placeholder in placeholder_list:
        for input_ in placeholder.inputs:
            if input_.copied_from is None:
                counter += 1
    return counter


def copy_combs_filter(
    combs_to_filter: list[list[list[bool]]], max_copy: int
) -> list[list[CompPlaceholder]]:
    ret = []
    for comb in combs_to_filter:
        temp_comb = []
        for composition in comb:
            if any(
                [
                    len([inp for inp in prev_comp.inputs if inp.copied_from])
                    >= max_copy - uniques(temp_comb[i:])
                    for i, prev_comp in enumerate(temp_comb)
                ]
            ):
                break  # The copies before this unique input and this unique input can't fit in memory
            # Create copies within the composition
            unique_inp_inds = [k for k in range(len(composition)) if composition[k]]
            if unique_inp_inds:  # we have unique inputs
                if (
                    len(composition) - len(unique_inp_inds) <= max_copy
                ):  # We can stay within the maximum allowed copies within this comp
                    input_placeholders = [
                        InputPlaceholder() for _ in range(len(composition))
                    ]
                    for ind, ip in enumerate(input_placeholders):
                        if ind not in unique_inp_inds:
                            parent = input_placeholders[random.choice(unique_inp_inds)]
                            if parent.copied_from is not None:
                                ip.copied_from = parent.copied_from
                                parent.copied_from.occurrence += 1
                            else:
                                ip.copied_from = parent
                                parent.occurrence += 1
                    comp_ph = CompPlaceholder(input_placeholders)
                    temp_comb.append(comp_ph)

            else:  # We want to combine the comps we have before it, if we can
                pool = []
                for i, t in enumerate(temp_comb):
                    # This candidate copy can still fit in the state tuple
                    valid = check_num_of_copies(temp_comb[i:], max_copy)
                    if len(t.inputs) == len(composition) and valid:
                        pool.append(t)
                if not pool:  # Nothing could be copied
                    break

                t = random.choice(pool)  # We select a random comp
                input_placeholders = []
                for inp in t.inputs:
                    if inp.copied_from is not None:
                        copy_inp = InputPlaceholder(copied_from=inp.copied_from)
                        inp.copied_from.occurrence += 1
                    else:  # inp is unique , None
                        copy_inp = InputPlaceholder(copied_from=inp)
                        inp.occurrence += 1
                    input_placeholders.append(copy_inp)

                if t.copied_from is not None:
                    t.copied_from.occurrence += 1
                    comp_ph = CompPlaceholder(
                        inputs=input_placeholders, copied_from=t.copied_from
                    )
                else:
                    t.occurrence += 1
                    comp_ph = CompPlaceholder(inputs=input_placeholders, copied_from=t)
                temp_comb.append(comp_ph)

        if len(temp_comb) == len(comb):
            ret.append(temp_comb)

    return ret


def possible_copy_combs(
    target_inputs: int,
    target_unique_inputs: int,
    max_split: int,  # Number of maximum splits a.k.a. the size of the state tuple
    use_all_perm: Union[int, bool] = True,
) -> list[list[list[bool]]]:
    inputs = (
        i < target_unique_inputs for i in range(target_inputs)
    )  # True represents unique input
    all_input_perms = set(permutations(inputs))
    if not use_all_perm and use_all_perm <= len(all_input_perms):
        all_input_perms = random.sample(all_input_perms, use_all_perm)
    ret = []
    for input_perm in all_input_perms:
        for n_split in range(max_split):
            for ind in combinations(range(1, len(input_perm)), n_split):
                comb = np.split(input_perm, ind)
                comb = [x.tolist() for x in comb if x.size != 0]
                # The unique inputs should be before the non unique
                # because we will use postorder comp building
                cool = True
                for x in comb:
                    if False in x:
                        ind = x.index(False)
                        head, tail = x[:ind], x[ind:]
                        tail = [not t for t in tail]
                        cool = cool and all(head) and all(tail)
                        if not cool:
                            break
                if cool:
                    ret.append(comb)
    return ret


def generate_all_possible_lens(all_comp_placeholders, target_comp_len):
    all_poss = []

    for comp_placeholders in all_comp_placeholders:
        io_of_top_comp = len(comp_placeholders)
        min_len_of_top_comp = io_of_top_comp - 1
        possible_len_of_all_unique_subcomps = []

        for comp_placeholder in comp_placeholders:
            if comp_placeholder.copied_from is None:
                lob = len(comp_placeholder.inputs) - 1
                hib = (
                    math.ceil(
                        (target_comp_len - min_len_of_top_comp)
                        / comp_placeholder.occurrence
                    )
                    + 1
                )
                possible_len_of_all_unique_subcomps.append(
                    (range(lob, hib), comp_placeholder.occurrence,)
                )
        top_range = range(min_len_of_top_comp, target_comp_len)

        multiplied_list = []
        original_list = []
        for range_, occ in possible_len_of_all_unique_subcomps:
            elem = [num * occ for num in range_]
            multiplied_list.append(elem)
            original_list.append(range_)

        multiplied_possible_combinations = list(
            itertools.product(top_range, *multiplied_list)
        )
        possible_combinations = list(itertools.product(top_range, *original_list))
        combinations_ = []
        for ind, candidate in enumerate(multiplied_possible_combinations):
            if (
                sum(candidate) == target_comp_len
                and (0 not in candidate or (candidate[1] != 0 and len(candidate) == 2))
                and candidate[0] >= len(comp_placeholders) - 1
            ):
                combinations_.append(possible_combinations[ind])
        if len(combinations_) != 0:
            all_poss.append(combinations_)
        else:
            all_poss.append(None)
    return all_poss


def populate_comp(comp_placeholder, leaf, used_zip_no_child, j):
    inputs = comp_placeholder.inputs
    if inputs[j].copied_from is None:
        if leaf.root_function.definition is zip_with:
            if not leaf.children:

                if inputs[j + 1].copied_from == inputs[j]:
                    temp = Input()
                    leaf.root_function.inputs = [temp, temp]
                    inputs[j].input = temp
                    leaf.root_function.operator = operator.mul
                elif inputs[j + 1].copied_from:
                    temp = inputs[j + 1].copied_from.input
                    new_input = Input()
                    leaf.root_function.inputs = [
                        new_input,
                        temp,
                    ]
                    inputs[j].input = new_input
                else:
                    new_input_1 = Input()
                    new_input_2 = Input()
                    leaf.root_function.inputs = [
                        new_input_1,
                        new_input_2,
                    ]

                    inputs[j].input = new_input_1
                    inputs[j + 1].input = new_input_2

                used_zip_no_child = True
            else:
                new_input = Input()
                leaf.root_function.inputs = [new_input]
                inputs[j].input = new_input  # No need to create a new one
        else:
            new_input = Input()
            leaf.root_function.inputs = [new_input]
            inputs[j].input = new_input
    elif leaf.root_function.definition is zip_with and not leaf.children:
        temp_1 = inputs[j].copied_from.input
        if inputs[j + 1].copied_from is not None:
            temp_2 = inputs[j + 1].copied_from.input
            if inputs[j].copied_from == inputs[j + 1].copied_from:
                leaf.root_function.operator = operator.mul
        else:
            temp_2 = Input()
            inputs[j + 1].input = temp_2
        leaf.root_function.inputs = [temp_1, temp_2]

        used_zip_no_child = True
    else:
        parent_inp = inputs[j].copied_from.input
        leaf.root_function.inputs = [parent_inp]

    if used_zip_no_child:
        used_zip_no_child = False
        j += 2
    else:
        j += 1

    return used_zip_no_child, j


def calculate_distribution(all_poss, num_samples):
    sum_curr = sum(len(curr) for curr in all_poss if curr is not None)
    sum_weighted = sum(sum_curr / len(curr) for curr in all_poss if curr is not None)
    samples_per_comp = [
        math.ceil(sum_curr / len(curr) * num_samples / sum_weighted / len(curr))
        if curr is not None
        else None
        for curr in all_poss
    ]

    return samples_per_comp


def compositions_from_combinations(
    comp_dict: Optional[CompDict],
    all_comp_placeholders: list[list[CompPlaceholder]],
    target_comp_len: int,
    num_comps: int,
    args: Arguments,
) -> Generator[Composition, None, None]:
    all_poss = generate_all_possible_lens(all_comp_placeholders, target_comp_len)
    samples_per_comp = calculate_distribution(all_poss, num_comps)
    curr_num_comps = 0
    curr_num_samples = 0

    while True:
        new_all_comp_placeholders = []
        new_all_poss = []

        for i, comp_placeholders in enumerate(all_comp_placeholders):
            if all_poss[i] is None or len(all_poss[i]) == 0:
                continue

            for _ in range(samples_per_comp[i]):
                for curr_lens in all_poss[i]:
                    tries = 0
                    while True:
                        if curr_lens[0] != 0:

                            top_comp = copy.deepcopy(
                                random.choice(
                                    comp_dict[curr_lens[0], len(comp_placeholders)]
                                )
                            )
                            comp_counter = 0
                            for leaf in top_comp.leaves:
                                if (
                                    leaf.root_function.definition is zip_with
                                    and not leaf.children
                                ):
                                    if (
                                        comp_placeholders[comp_counter]
                                        == comp_placeholders[
                                            comp_counter + 1
                                        ].copied_from
                                    ):
                                        leaf.root_function.operator = operator.mul
                                    comp_counter += 2
                                else:
                                    comp_counter += 1

                        bottom_comps = []
                        comp_placeholders_copy = copy.deepcopy(comp_placeholders)

                        k = 1
                        for comp_placeholder in comp_placeholders_copy:
                            if comp_placeholder.copied_from is None:
                                curr_len = curr_lens[k]
                                key = (
                                    curr_len,
                                    len(comp_placeholder.inputs),
                                    False,
                                )

                                comp = copy.deepcopy(
                                    random.choice(comp_dict[key])
                                ).copy_with_new_ids()

                                if (
                                    curr_lens[0] == 0
                                ):  # We don't want to extend the top of the comp
                                    top_comp = comp

                                # Populate comp
                                j = 0
                                used_zip_no_child = False
                                for leaf in comp.leaves:
                                    used_zip_no_child, j = populate_comp(
                                        comp_placeholder, leaf, used_zip_no_child, j
                                    )

                                bottom_comps.append(comp)
                                comp_placeholder.composition = comp
                                k += 1
                            else:
                                # Copy the comp from its parent
                                bottom_comps.append(
                                    comp_placeholder.copied_from.composition
                                )

                        if curr_lens[0] != 0:
                            # noinspection PyUnboundLocalVariable
                            top_comp.extend_leaves(
                                bottom_comps,
                                [None] * len(bottom_comps),  # TODO: not None
                            )

                        try:
                            can_increment = True
                            for _ in range(args.num_samples_per_comp):
                                try:
                                    if not args.is_test:
                                        samples = top_comp.as_samples(args.io)
                                        yield samples
                                    else:
                                        samples = top_comp.as_samples(args.io)
                                        yield [
                                            {
                                                "input": samples[0]["input"],
                                                "output": samples[0]["output"],
                                                "solution": [
                                                    sample["next_transformation"]
                                                    for sample in samples
                                                ],
                                            }
                                        ]
                                    curr_num_samples += 1

                                    if (
                                        curr_num_samples
                                        >= num_comps * args.num_samples_per_comp
                                    ):
                                        return
                                except UnimprovableInputError:
                                    if can_increment:
                                        can_increment = False

                            if can_increment:
                                curr_num_comps += 1

                            if comp_placeholders not in new_all_comp_placeholders:
                                new_all_comp_placeholders.append(comp_placeholders)
                                new_all_poss.append(all_poss[i])

                            break
                        except FullStateTupleError:
                            tries += 1
                            if tries == 10:
                                break

        samples_per_comp = calculate_distribution(
            new_all_poss, num_comps - curr_num_comps
        )
        all_poss = new_all_poss
        all_comp_placeholders = new_all_comp_placeholders


def generate_copy_comps(
    comp_dict: CompDict, number: int, args: Arguments
) -> Generator[Composition, None, None]:
    one_input = args.inputs == args.unique_inputs == 1
    num_copy = args.inputs - args.unique_inputs
    no_copy = num_copy == 0

    if one_input or no_copy:
        raise ValueError(
            f"Called generate_copy_comp with invalid args: {one_input=} {no_copy=}"
        )

    if config.STATE_TUPLE_LENGTH_HIB * args.unique_inputs < args.inputs:
        raise ValueError(
            f"Called generate_copy_comp with invalid args: {args.inputs=} "
            f"{args.unique_inputs=} {config.STATE_TUPLE_LENGTH_HIB=}"
        )

    possible_combs = possible_copy_combs(
        args.inputs,
        args.unique_inputs,
        max_split=config.STATE_TUPLE_LENGTH_HIB,
        use_all_perm=True,
    )

    all_comp_placeholders = copy_combs_filter(
        possible_combs, max_copy=config.STATE_TUPLE_LENGTH_HIB
    )

    yield from compositions_from_combinations(
        comp_dict, all_comp_placeholders, args.functions, number, args
    )


# noinspection PyUnresolvedReferences
class DataWorker(QObject):
    bar_advanced = pyqtSignal(int)
    start_signal = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, args: Arguments):
        super().__init__()
        self.args = args
        self.start_signal.connect(self.process, Qt.QueuedConnection)

    @pyqtSlot()
    def process(self):
        with open(self.args.filename, "w") as f:
            for i, sample_list in enumerate(sample_generator(self.args)):
                for sample in sample_list:
                    json.dump(sample, f)
                    f.write("\n")

                self.bar_advanced.emit(
                    math.ceil(
                        100 * i / (self.args.number * self.args.num_samples_per_comp)
                    )
                )

                # if self.shutdown:
                #     self.finished.emit()
                #     return
            self.finished.emit()


if __name__ == "__main__":
    args_ = Arguments(
        number=2000,
        filename="alma.txt",
        functions=2,
        io=1,
        inputs=1,
        unique_inputs=1,
        num_samples_per_comp=1,
        is_test=True,
    )
    worker = DataWorker(args_)
    worker.process()
