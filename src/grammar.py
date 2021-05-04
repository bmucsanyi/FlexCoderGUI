import operator

from nltk import CFG, RecursiveDescentParser
from nltk.parse.generate import generate

from src.func_impl import *

GRAMMAR = CFG.fromstring(
    """
    S -> array_function | numeric_function
    array_function -> "sort(""array)" | "take("pos", array)" | "drop("pos", array)"
    array_function -> "reverse_func(""array)" | "map_func("num_lambda_unary", array)"
    array_function -> "filter_func("bool_lambda_unary", array)" | "zip_with("num_lambda_binary", array, array)"
    numeric_function -> "max(""array)" | "min(""array)" | "sum(""array)" | "length(""array)"
    num -> neg | "0" | pos
    less_than_minus_one -> "-8" | "-7" | "-6" | "-5" | "-4" | "-3" | "-2"
    neg -> less_than_minus_one | "-1"
    pos -> "1" | greater_than_one
    greater_than_one ->  "2" | "3" | "4" | "5" | "6" | "7" | "8"
    bool_lambda_unary -> "lambda x: x" bool_unary_operator num | "lambda x: x" mod
    bool_unary_operator -> " == " | " < " | " > "
    mod -> " % " greater_than_one " == 0"
    num_lambda_unary -> "lambda x: x" num_unary_operator
    num_lambda_binary -> "lambda x, y: x"num_binary_operator"y" | num_binary_function
    num_unary_operator -> " * " mul_num | " // " div_num | " + " pos | " - " pos | " % " greater_than_one
    num_binary_operator -> " * " | " + " | " - "
    num_binary_function -> "max" | "min"
    div_num -> less_than_minus_one | greater_than_one
    mul_num -> neg | "0" | greater_than_one
    """
)

MAX_NUM = 8

MIN_NUM = -8

DEFINITIONS = [
    "max",
    "min",
    "sum",
    "length",
    "sort",
    "take",
    "drop",
    "reverse_func",
    "map_func",
    "filter_func",
    "zip_with",
    "copy_state_tuple",
]

ABBREVIATION_DICT = {
    "max": "max",
    "min": "min",
    "sum": "sum",
    "length": "len",
    "sort": "sort",
    "take": "take",
    "drop": "drop",
    "reverse_func": "rev",
    "map_func": "map",
    "filter_func": "filter",
    "zip_with": "zip",
}

BOOL_LAMBDA_OPERATORS = ["==", "<", ">", "%"]

BOOL_LAMBDA_NUMBERS = [str(r) for r in range(MIN_NUM, MAX_NUM + 1)]

NUM_LAMBDA_OPERATORS = ["+", "-", "*", "//", "%", "max", "min"]

NUM_LAMBDA_NUMBERS = [str(r) for r in range(MIN_NUM, MAX_NUM + 1)]

TAKE_DROP_NUMBERS = [str(r) for r in range(1, MAX_NUM + 1)]

TOKEN_MAP = {
    "array_function": "function",
    "numeric_function": "function",
    "neg": "number",
    "pos": "number",
    "num": "number",
    "greater_than_one": "number",
    "less_than_minus_one": "number",
    "div_num": "number",
    "mul_num": "number",
    "bool_unary_operator": "operator",
    "num_binary_operator": "operator",
    "num_unary_operator": "operator",
    "num_binary_function": "operator",
    "mod": "operator",
}

FUNC_DICT = {
    "max": max,
    "min": min,
    "sum": sum,
    "length": length,
    "sort": sort,
    "take": take,
    "drop": drop,
    "reverse_func": reverse_func,
    "map_func": map_func,
    "filter_func": filter_func,
    "zip_with": zip_with,
}

OPERATOR_DICT = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "//": operator.floordiv,
    "%": operator.mod,
    "==": operator.eq,
    "<": operator.lt,
    ">": operator.gt,
    "max": max,
    "min": min,
}

REVERSE_OPERATOR_DICT = {v: k for k, v in OPERATOR_DICT.items()}


def generate_code(depth: int = 9, separator: str = "") -> set[str]:
    """Generates function compositions using NLTK.

    Args:
        depth: Optional variable,
            maximal depth of the generated tree of function compositions.

        separator: The separator string of tokens in the generated derivations.

    Returns:
        A set containing strings representing function compositions.
    """

    return {
        separator.join(product).strip() for product in generate(GRAMMAR, depth=depth)
    }


ALL_FUNCTION_STRINGS = sorted(list(generate_code(depth=200, separator="!")))


def get_parser(grammar: CFG, separator: str = "!") -> Callable[[str], dict]:
    """Returns a parser that maps string representations of parameterized
    functions to a dictionary which contains each part of the function.

    Args:
        grammar: The grammar from which the functions can be derived.
        separator: String separating the tokens of the grammar in the
          parameterized functions.
    """

    def parser(sample: str) -> dict:
        tokenized_sample = [token for token in sample.split(separator) if token != " "]
        rd_parser = RecursiveDescentParser(grammar)
        res = rd_parser.parse(tokenized_sample)
        pos = list(res)[0].pos()
        res = {k: v for v, k in pos[::-1]}
        res = {
            TOKEN_MAP[k]: v.strip("(").strip() for k, v in res.items() if k in TOKEN_MAP
        }

        return res

    return parser


PARSER = get_parser(GRAMMAR)


def main():
    pass


if __name__ == "__main__":
    main()
