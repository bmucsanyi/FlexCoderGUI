import itertools

import torch
from scipy.stats.mstats import gmean

from src.composition import *
from src.dataset import FlexDataset
from src.function import ALL_FUNCTIONS
from src.grammar import (
    DEFINITIONS,
    NUM_LAMBDA_NUMBERS,
    NUM_LAMBDA_OPERATORS,
    BOOL_LAMBDA_NUMBERS,
    BOOL_LAMBDA_OPERATORS,
    TAKE_DROP_NUMBERS,
    FUNC_DICT,
    OPERATOR_DICT,
)
from src.model import FlexNet

# from src.cache import cache

_FUNC_DICT = {**FUNC_DICT, "copy_state_tuple": copy_state_tuple}

_ALL_FUNCTIONS = ALL_FUNCTIONS + [
    Function(definition=copy_state_tuple, operator=None, number=None)
]  # TODO: move copy

DEFINITION_INDEX = {
    _FUNC_DICT[definition]: ind for ind, definition in enumerate(DEFINITIONS)
}
NUM_LAMBDA_NUMBERS_INDEX = {int(num): ind for ind, num in enumerate(NUM_LAMBDA_NUMBERS)}
NUM_LAMBDA_OPERATORS_INDEX = {
    OPERATOR_DICT[op]: ind for ind, op in enumerate(NUM_LAMBDA_OPERATORS)
}
BOOL_LAMBDA_NUMBERS_INDEX = {
    int(num): ind for ind, num in enumerate(BOOL_LAMBDA_NUMBERS)
}
BOOL_LAMBDA_OPERATORS_INDEX = {
    OPERATOR_DICT[op]: ind for ind, op in enumerate(BOOL_LAMBDA_OPERATORS)
}
TAKE_DROP_NUMBERS_INDEX = {int(num): ind for ind, num in enumerate(TAKE_DROP_NUMBERS)}

# weights appear in the order of the extractor functions
weight_dict = {
    max: (1,),  # definition
    min: (1,),  # definition
    sum: (1,),  # definition
    length: (1,),  # definition
    sort: (0.8, 0.2),  # definition, index
    take: (0.6, 0.2, 0.2),  # definition, index, number
    drop: (0.6, 0.2, 0.2),  # definition, index, number
    reverse_func: (0.8, 0.2),  # definition, index
    map_func: (0.5, 0.2, 0.15, 0.15),  # definition, index, operator, number
    filter_func: (0.5, 0.2, 0.15, 0.15),  # definition, index, operator, number
    zip_with: (0.5, 0.2, 0.2, 0.1),  # definition, index, index, operator
    copy_state_tuple: (0.8, 0.2),  # definition, index
}


class ModelFacade:
    def __init__(self, weights) -> None:
        self.model = FlexNet.load_from_checkpoint(weights)

    # @cache.cache()
    def predict(self, state_tuple, target):
        inp = [x[0][0] for x in state_tuple]  # This might only work for single io
        # print(inp)
        inp, inp_lenghts = FlexDataset.to_processed_tensor(inp, is_input=True)
        inp_lenghts = [[x] for x in inp_lenghts]
        inp = inp.unsqueeze(0)

        out, out_lenghts = FlexDataset.to_processed_tensor(target, is_input=False)
        out = out.unsqueeze(0)

        if any(x[0] == 0 for x in inp_lenghts):  # TODO ez is lehet fos
            return torch.zeros(7, 17)

        with torch.no_grad():
            output = self.model(
                inp,
                input_lengths=inp_lenghts,
                output=out,
                output_lengths=[out_lenghts],
            )

        return [
            torch.softmax(o, dim=-1)[0] if ind != 1 else torch.sigmoid(o)[0]
            for ind, o in enumerate(output)
        ]
        # return [torch.sigmoid(o)[0] for o in output]


@dataclass
class SearchNode:
    function: Function
    indices: list[int]
    state_tuple: Optional[tuple] = None
    rank: Optional[float] = None
    parent: Optional["SearchNode"] = None


@dataclass
class RankAssigner:
    aggregator_func: Callable = gmean

    def _definition_weight(self, node: SearchNode, weights):
        return weights[0][DEFINITION_INDEX[node.function.definition]].item()

    def _first_index(self, node: SearchNode, weights):
        return weights[1][node.indices[0]].item()

    def _second_index(self, node: SearchNode, weights):
        if len(node.indices) == 2:
            return weights[1][node.indices[1]].item()

    def _bool_lambda_op_weight(self, node: SearchNode, weights):
        if node.function.definition is filter_func:
            return weights[2][
                BOOL_LAMBDA_OPERATORS_INDEX[node.function.operator]
            ].item()

    def _bool_lambda_num_weight(self, node: SearchNode, weights):
        if node.function.definition is filter_func:
            return weights[3][BOOL_LAMBDA_NUMBERS_INDEX[node.function.number]].item()

    def _num_lambda_op_weight(self, node: SearchNode, weights):
        if node.function.definition in (map_func, zip_with):
            return weights[4][NUM_LAMBDA_OPERATORS_INDEX[node.function.operator]].item()

    def _num_lambda_num_weight(self, node: SearchNode, weights):
        if node.function.definition is map_func:
            return weights[5][NUM_LAMBDA_NUMBERS_INDEX[node.function.number]].item()

    def _take_drop_num_weight(self, node: SearchNode, weights):
        if node.function.definition in (take, drop):
            return weights[6][TAKE_DROP_NUMBERS_INDEX[node.function.number]].item()

    def __post_init__(self):
        self.extractors = [
            self._definition_weight,
            self._first_index,
            self._second_index,
            self._num_lambda_op_weight,
            self._num_lambda_num_weight,
            self._bool_lambda_op_weight,
            self._bool_lambda_num_weight,
            self._take_drop_num_weight,
        ]

    def calculate_rank(self, node: SearchNode, weights):
        individual_weights = [
            res for func in self.extractors if (res := func(node, weights)) is not None
        ]

        return self.aggregator_func(node, individual_weights)


def aggregator(node: SearchNode, predicted_weights):
    func = node.function.definition
    weightings = weight_dict[func]

    nom = sum(np.log(x) * w for x, w in zip(predicted_weights, weightings))
    denom = sum(weightings)

    return np.exp(nom / denom)


wa = RankAssigner(aggregator)
model = ModelFacade("model_checkpoints/softmax_1M_lenght_6_io_1.ckpt")


def generate_children_non_zipwith(function: Function, parent: SearchNode, weights):
    if function.definition in (max, min, length, sum) and len(parent.state_tuple) != 1:
        return

    for ind in range(len(parent.state_tuple)):

        node = SearchNode(function=function, indices=[ind], parent=parent)

        state_tuple = node.state_tuple = function.eval_on_state_tuple(
            parent.state_tuple, node.indices
        )

        if state_tuple is None:
            continue

        node.state_tuple = state_tuple
        node.rank = parent.rank + np.log(wa.calculate_rank(node, weights))

        yield node


def generate_children_zipwith(function: Function, parent: SearchNode, weights):
    if function.operator is operator.sub:
        iterator = itertools.permutations(
            range(len(parent.state_tuple)), 2
        )  # TODO: Vágás után kiterjesztés előtt bs-ben
    else:
        iterator = itertools.combinations(range(len(parent.state_tuple)), 2)

    for ind in iterator:
        node = SearchNode(function=function, indices=list(ind), parent=parent)
        state_tuple = node.state_tuple = function.eval_on_state_tuple(
            parent.state_tuple, node.indices
        )

        if state_tuple is None:
            continue

        node.state_tuple = state_tuple
        node.rank = parent.rank + np.log(wa.calculate_rank(node, weights))

        yield node


def generate_children(parent: SearchNode, target):
    weights = model.predict(parent.state_tuple, target)
    for function in _ALL_FUNCTIONS:
        # TODO: itt lehetne szűrni a lehetséges függvényeket az elvárt output típus alapján.
        #  Pl ha tömb akkor fölösek a max,min és barátaik
        if function.definition is not zip_with:
            yield from generate_children_non_zipwith(function, parent, weights)
        elif function.definition is zip_with:
            yield from generate_children_zipwith(function, parent, weights)
        else:
            raise ValueError("asd")


counter = 0


def write_path(node: SearchNode, solution):
    if node.function is not None:
        solution.append(node)
        # print(str(node.function), node.indices)

    if node.parent is not None:
        write_path(node.parent, solution)


def manage_state_tuple(node: SearchNode, state_tuple, comp: Composition):
    result_ind = min(node.indices)
    indices_to_remove = [ind for ind in node.indices if ind != result_ind]

    state_tuple[result_ind] = comp

    for rem in indices_to_remove:
        del state_tuple[rem]


def create_solution_comp(solution: list[SearchNode], inp: tuple) -> Composition:
    state_tuple = [i[0][0] for i in inp]
    for node in solution:
        if node.function.definition is copy_state_tuple:
            ind = node.indices[0]
            state_tuple.append(copy.deepcopy(state_tuple[ind]))
        else:
            if all(isinstance(state_tuple[ind], list) for ind in node.indices):
                func = copy.deepcopy(node.function)
                func.inputs = [Input(state_tuple[ind]) for ind in node.indices]
                comp = Composition(func)
                manage_state_tuple(node, state_tuple, comp)

            elif all(isinstance(state_tuple[ind], Composition) for ind in node.indices):
                branches_to_merge = [state_tuple[ind] for ind in node.indices]
                comp = Composition.from_composition(node.function, *branches_to_merge)
                manage_state_tuple(node, state_tuple, comp)

            else:
                func = copy.deepcopy(node.function)
                branch = None
                for ind in node.indices:
                    if isinstance(state_tuple[ind], list):
                        func.inputs = [Input(state_tuple[ind])]
                    else:
                        branch = state_tuple[ind]
                comp = Composition.from_composition(func, branch)
                manage_state_tuple(node, state_tuple, comp)

    return state_tuple[0]


def beam_search(
    inp: tuple,
    beam_size: int,
    max_length,
    target: OutputType,
) -> Optional[SearchNode]:
    def transform_target(target):
        # This just makes the api a bit more user friendly
        val = (
            isinstance(target, int)
            or isinstance(target, list)
            and len(target) > 0
            and isinstance(target[0], int)
            or target == []
        )

        if val:
            return [target]
        return target

    transformed_target = transform_target(target)

    def is_leaf(node: SearchNode):

        if any(any(isinstance(y, int) for y in x[0]) for x in node.state_tuple):
            return True

        if isinstance(transformed_target[0], list):
            min_list_lengths = [len(l) for l in transformed_target]
            # print("-----")
            # print(a)
            # print([x[0] for x in node.state_tuple])
            # print([list(itertools.chain.from_iterable(x)) for x in node.state_tuple])

            # print([[(len(y), le) for y, le in zip(x[0], a)] for x in node.state_tuple])
            # If for any list is shorter then it has to be for the target -> cant be solved -> leaf
            res = any(
                [
                    any(
                        len(array) < min_list_length
                        for array, min_list_length in zip(x[0], min_list_lengths)
                    )
                    for x in node.state_tuple
                ]
            )
            if res:
                global counter
                counter += 1

            return res

        return False

    def is_solution(node: SearchNode) -> bool:
        # print(node.state_tuple[0][0])
        # TODO: Allow one length arrays and number to be solutions
        return len(node.state_tuple) == 1 and (
            node.state_tuple[0][0] == transformed_target
            or node.state_tuple[0][0][0] == transformed_target[0][0]
        )

    def make_stat(node: SearchNode, depth, inp: tuple):
        if node is not None:
            solution = []
            write_path(node, solution)
            solution = list(reversed(solution))
            buffer = []
            for node in solution:
                if node.function is not None:
                    buffer.append((str(node.function), node.indices))
            solution_comp = create_solution_comp(solution, inp)
        else:
            buffer = None
            solution_comp = None
        return {
            "program": buffer,
            "depth": depth,
            "composition": solution_comp,
        }

    root = SearchNode(None, indices=None, state_tuple=inp, rank=1, parent=None,)
    for iteration in range(3):
        beam = beam_size * (2 ** iteration)
        nodes = [root]

        for depth in range(max_length):
            new_nodes = itertools.chain.from_iterable(
                [list(generate_children(node, transformed_target)) for node in nodes]
            )
            sorted_nodes = sorted(new_nodes, key=lambda x: x.rank, reverse=True)

            nodes.clear()
            for node in sorted_nodes:
                if is_solution(node):
                    return make_stat(node, depth, inp)
                if not is_leaf(node):
                    nodes.append(node)
                    if len(nodes) == beam:
                        break

    return make_stat(None, None, inp)


if __name__ == "__main__":
    state_tuple = (
        [
            [
                [
                    -204,
                    -43,
                    -7,
                    9,
                    -35,
                    6,
                    -7,
                    5,
                    -153,
                    -71,
                    -52,
                    -7,
                    -9,
                    -168,
                    -7,
                    -129,
                    9,
                    -8,
                ]
            ]
        ],
        [[[-2, -4, -7, 9, 3, 6, -7, 5, -2, -2, -3, -7, -9, -1, -7, 2, 9, -8]]],
    )
    target = [755]

    res = model.predict(state_tuple, [target])
    for ind, r in enumerate(res):
        print(ind, r)

    res = beam_search(state_tuple, 150, 8, target=[target])

    if res is None:
        print("FUCK")
    else:
        write_path(res)
        print(res.state_tuple)
    print(counter)
