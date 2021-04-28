from dataclasses import dataclass
from itertools import cycle
from typing import TYPE_CHECKING, Callable, Generator, Union

import pydotplus as pydot
import torch

import config

if TYPE_CHECKING:
    from src.composition import Composition
    from src.input_ import Input
    from src.generate_utils import CompPlaceholder


def get_divisors(num: int) -> Generator[int, None, None]:
    num = abs(num)
    for i in range(num, 0, -1):
        if num % i == 0:
            yield i


def calculate_distribution(result: int) -> tuple[int, int]:
    for i in get_divisors(result):
        if i <= 8:
            return result // i, i


def get_id_generator() -> Callable:
    counter = 0

    def id_generator() -> int:
        nonlocal counter
        counter += 1
        return counter - 1

    return id_generator


generator = get_id_generator()


@dataclass
class CompOccurrence:
    comp: Union["Composition", "Input"]
    occurrences: int


def visualize(comp: "Composition", filename: str = "output.png") -> None:
    graph = pydot.Dot("my_graph", graph_type="graph", bgcolor="#303030")
    graph.set_node_defaults(color='white', fontname='Courier', fontsize='12', fontcolor='white')

    if len(comp) > 1:
        comp_occurrences = [CompOccurrence(comp=comp, occurrences=1)]
        visualize_helper(comp, graph, comp_occurrences)
    else:
        graph.add_node(pydot.Node(str(comp.root_function).replace(":", ";")))
    graph.write_png(filename)


def visualize_helper(comp: "Composition", graph, comp_occurrences):
    def get_comp_occurrence(curr_comp: Union["Composition", "Input"]):
        for co in comp_occurrences:
            if co.comp == curr_comp:
                return co
        return None

    def get_function_string(comp_: "Composition", is_child: bool):
        root_function = comp_.root_function
        input_occurrences = []
        for i, input_ in enumerate(root_function.inputs):
            ch_inp_occurrence = get_comp_occurrence(input_)
            if not ch_inp_occurrence:
                comp_occurrences.append(CompOccurrence(comp=input_, occurrences=1))
                input_occurrences.append(1)
            else:
                if is_child:
                    ch_inp_occurrence.occurrences += 1
                input_occurrences.append(ch_inp_occurrence.occurrences)

        function_string = str(root_function)

        placeholder_dict = {}
        counter = config.INPUT_LOB - 1
        for input_, input_occurrence in zip(root_function.inputs, input_occurrences):
            placeholder_dict[counter] = input_.data
            function_string = function_string.replace(
                str(input_.data), f"{counter} v{input_.id}.{input_occurrence}", 1
            )
            counter -= 1

        for counter in placeholder_dict:
            function_string = function_string.replace(
                str(counter), str(placeholder_dict[counter]), 1
            )

        return function_string

    par_occurrence = get_comp_occurrence(comp).occurrences
    for child in comp.children:
        ch_comp_occurrence = get_comp_occurrence(child)
        if not ch_comp_occurrence:
            comp_occurrences.append(CompOccurrence(comp=child, occurrences=1))
            ch_occurrence = 1
        else:
            ch_comp_occurrence.occurrences += 1
            ch_occurrence = ch_comp_occurrence.occurrences

        comp_string = get_function_string(comp, is_child=False)
        child_string = get_function_string(child, is_child=True)

        first_str = f"{comp_string} v{comp.id}.{par_occurrence}"
        second_str = f"{child_string} v{child.id}.{ch_occurrence}"
        graph.add_edge(pydot.Edge(first_str, second_str, color="white"))

        visualize_helper(child, graph, comp_occurrences)


def visualize_placeholders(
    placeholder_list: list["CompPlaceholder"], file_name: str = "placeholder.png"
):
    colors = cycle(("lightblue", "red", "forestgreen", "orange", "yellow", "red"))
    color_pairs = {}
    selected_color = next(colors)
    graph = pydot.Dot("my_graph", graph_type="digraph", bgcolor="white")
    for i, x in enumerate(placeholder_list):
        if x.copied_from is not None:
            color_pairs[id(x)] = color_pairs[id(x.copied_from)]
            graph.add_edge(
                pydot.Edge(
                    f"{x.copied_from.__class__} {id(x.copied_from)}",
                    f"{x.__class__} {id(x)}",
                    color=color_pairs[id(x)],
                    style="dashed",
                )
            )
            graph.add_node(
                pydot.Node(
                    f"{x.__class__} {id(x)}",
                    style="filled",
                    fillcolor=color_pairs[id(x)],
                )
            )
        else:
            color_pairs[id(x)] = selected_color
            graph.add_node(
                pydot.Node(
                    f"{x.__class__} {id(x)}",
                    style="filled",
                    fillcolor=color_pairs[id(x)],
                )
            )
            selected_color = next(colors)
        for y in x.inputs:
            if y.copied_from is not None:
                color_pairs[id(y)] = color_pairs[id(y.copied_from)]
                graph.add_edge(
                    pydot.Edge(
                        f"{y.copied_from.__class__} {id(y.copied_from)}",
                        f"{y.__class__} {id(y)}",
                        color=color_pairs[id(y)],
                        style="dashed",
                    )
                )
                graph.add_node(
                    pydot.Node(
                        f"{y.__class__} {id(y)}",
                        style="filled",
                        fillcolor=color_pairs[id(y)],
                    )
                )
            else:
                color_pairs[id(y)] = selected_color
                graph.add_node(
                    pydot.Node(
                        f"{y.__class__} {id(y)}",
                        style="filled",
                        fillcolor=color_pairs[id(y)],
                    )
                )
                selected_color = next(colors)
            graph.add_edge(
                pydot.Edge(
                    f"{x.__class__} {id(x)}",
                    f"{y.__class__} {id(y)}",
                    color=color_pairs[id(x)],
                    style="dashed",
                )
            )
    graph.write_png(file_name)


def symmetric_floordiv(a: int, b: int):
    if a * b < 0:
        return int(-(abs(a) // abs(b)))
    else:
        return int(a // b)


def collate_fn(batch: list[tuple]) -> tuple:
    inputs = []
    input_lengths = [[] for _ in range(config.STATE_TUPLE_LENGTH_HIB)]
    outputs = []
    output_lengths = []
    definitions = []
    indices = []
    bool_lambda_ops = []
    bool_lambda_nums = []
    num_lambda_ops = []
    num_lambda_nums = []
    take_drop_nums = []

    for elem in batch:
        inputs.append(elem[0][0][0])

        for ind, length in enumerate(elem[0][0][1]):
            input_lengths[ind].append(length)

        outputs.append(elem[0][1][0])
        output_lengths.append(elem[0][1][1])

        definitions.append(elem[1][0])
        indices.append(elem[1][1])
        bool_lambda_ops.append(elem[1][2])
        bool_lambda_nums.append(elem[1][3])
        num_lambda_ops.append(elem[1][4])
        num_lambda_nums.append(elem[1][5])
        take_drop_nums.append(elem[1][6])

    return (
        ((torch.stack(inputs), input_lengths), (torch.stack(outputs), output_lengths)),
        (
            torch.stack(definitions),
            torch.stack(indices),
            torch.stack(bool_lambda_ops),
            torch.stack(bool_lambda_nums),
            torch.stack(num_lambda_ops),
            torch.stack(num_lambda_nums),
            torch.stack(take_drop_nums),
        ),
    )


def print_batch(batch: tuple) -> None:
    for i in range(2):
        print(batch[0][i][0].shape)
        print(len(batch[0][i][1]))

    for i in range(7):
        print(batch[1][i].shape)
