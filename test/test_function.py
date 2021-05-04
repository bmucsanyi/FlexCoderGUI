import copy
import operator

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.composition import Composition
from src.func_impl import *
from src.function import REVERSE_OPERATOR_DICT, Function, Input
from src.grammar import ALL_FUNCTION_STRINGS, GRAMMAR, get_parser


# from src.utils import visualize


class TestFunction:
    def test_from_dict(self):
        parser = get_parser(GRAMMAR)
        for sample in ALL_FUNCTION_STRINGS:
            parsed = parser(sample)
            function = Function.from_dict(parsed)

            assert function.definition.__name__ == parsed["function"]
            if "number" in parsed:
                assert function.number == int(parsed["number"])
            if "operator" in parsed:
                assert REVERSE_OPERATOR_DICT[function.operator] == parsed["operator"]

    @settings(deadline=None, max_examples=10)
    @given(st.lists(st.integers(), min_size=1), st.lists(st.integers(), min_size=1))
    def test_eval_str_1_io_zip_with(self, input_1, input_2):
        zip_withs = Function.get_buckets()[zip_with]

        for function in zip_withs:
            function.inputs = [Input(input_1), Input(input_2)]

            assert function.eval() == eval(str(function))

    @settings(deadline=None, max_examples=10)
    @given(st.lists(st.integers(), min_size=1))
    def test_eval_str_1_io_other(self, input_):
        buckets = Function.get_buckets()
        del buckets[zip_with]

        for bucket in buckets:
            for function in buckets[bucket]:
                function.inputs = [Input(input_)]
                assert function.eval() == eval(str(function))

    @settings(deadline=None, max_examples=10)
    @given(st.lists(st.lists(st.integers(), min_size=1), min_size=1))
    def test_eval_more_io_other(self, input_):
        buckets = Function.get_buckets()
        del buckets[zip_with]

        for bucket in buckets:
            self._more_io_helper(buckets[bucket], input_)

    @settings(deadline=None, max_examples=10)
    @given(
        st.lists(st.lists(st.integers(), min_size=1), min_size=1),
        st.lists(st.lists(st.integers(), min_size=1), min_size=1),
    )
    def test_eval_more_io_zip_with(self, input_1, input_2):
        assume(len(input_1) == len(input_2))
        zip_withs = Function.get_buckets()[zip_with]

        self._more_io_helper(zip_withs, input_1, input_2)

    @staticmethod
    def _more_io_helper(bucket, input_1, input_2=None):
        for function in bucket:
            function.inputs = (
                [Input(input_1), Input(input_2)]
                if input_2 is not None
                else [Input(input_1)]
            )
            result_1 = function.eval()
            function.inputs = []
            if input_2 is not None:
                result_2 = [function.eval([a, b]) for a, b in zip(input_1, input_2)]
            else:
                result_2 = [function.eval([a]) for a in input_1]

            assert result_1 == result_2


class TestComposition:
    def setup_method(self):
        map_func_ = Function.from_dict(
            {"function": "map_func", "operator": "*", "number": "2"}
        )

        take_ = Function.from_dict({"function": "take", "number": "2"})
        reverse_func_ = Function.from_dict({"function": "reverse_func"})

        zip_with_1 = Function.from_dict({"function": "zip_with", "operator": "*"})

        filter_func_ = Function.from_dict(
            {"function": "filter_func", "operator": ">", "number": "7"}
        )
        zip_with_2 = Function.from_dict({"function": "zip_with", "operator": "+"})
        sum_ = Function.from_dict(
            {"function": "sum"}
        )  # not max, as nonemptiness is not guaranteed

        map_node = Composition(map_func_)
        map_node = Composition.from_composition(take_, map_node)  # change of order
        self.take_node = map_node.children[0]

        self.reverse_node = Composition(reverse_func_)
        self.zip_with_1_node = Composition.from_composition(
            zip_with_1, self.reverse_node
        )
        filter_node = Composition.from_composition(filter_func_, self.zip_with_1_node)
        zip_with_2_node = Composition.from_composition(
            zip_with_2, map_node, filter_node
        )

        self.composition = Composition.from_composition(sum_, zip_with_2_node)
        # visualize(self.composition, "test_comp.png")

        map_func__ = Function.from_dict(
            {"function": "map_func", "operator": "*", "number": "2"}
        )

        take__ = Function.from_dict({"function": "take", "number": "2"})
        reverse_func__ = Function.from_dict({"function": "reverse_func"})

        zip_with_1__ = Function.from_dict({"function": "zip_with", "operator": "*"})

        filter_func__ = Function.from_dict(
            {"function": "filter_func", "operator": ">", "number": "7"}
        )
        zip_with_2__ = Function.from_dict({"function": "zip_with", "operator": "+"})

        map_node__ = Composition(map_func__)
        map_node__ = Composition.from_composition(take__, map_node__)  # change of order
        take_node__ = map_node__.children[0]

        reverse_node__ = Composition(reverse_func__)
        take_node__._add_child(reverse_node__)

        zip_with_1_node__ = Composition.from_composition(zip_with_1__, reverse_node__)
        filter_node__ = Composition.from_composition(filter_func__, zip_with_1_node__)
        self.composition2 = Composition.from_composition(
            zip_with_2__, map_node__, filter_node__
        )
        # visualize(self.composition2, "test_comp2.png")

    @settings(deadline=None, max_examples=10)
    @given(
        st.lists(st.integers(), min_size=1),
        st.lists(st.integers(), min_size=1),
        st.lists(st.integers(), min_size=1),
    )
    def test_eval_str_leaves_1_io(self, input_1, input_2, input_3):
        leaves: list[Composition] = self.composition.leaves
        assert len(leaves) == 3

        for leaf in leaves:
            assert leaf.root_function.definition in {take, reverse_func, zip_with}

        for leaf, input_ in zip(leaves, (input_1, input_2, input_3)):
            leaf.root_function.inputs = [Input(input_)]

        assert self.composition.eval() == eval(str(self.composition))

    def test_eval_more_io(self):
        input_1 = Input([[9, 6, 7, 3, 4], [8, 2, 1], [7, 7, 7, 7], [0, 1, 0, 1]])
        input_2 = Input(
            [[9, 4, 9, 4, 3], [6, 9, 14, 3, 2], [1, 99, 3], [0, 6, 40, 50, 50, 50]]
        )
        input_3 = Input([[1, 5, 3, 2, 9], [1, 1, 1, 1], [6, 6, 6, 7], [12, 16, 20]])

        for leaf, input_ in zip(self.composition.leaves, (input_1, input_2, input_3)):
            leaf.root_function.inputs = [input_]

        assert self.composition.eval() == [119, 43, 640, 578]

    def test_mul_floordiv_contractions(self):
        map_func_1 = Function.from_dict(
            {"function": "map_func", "number": 6, "operator": "*"}
        )
        map_func_2 = Function.from_dict(
            {"function": "map_func", "number": 2, "operator": "//"}
        )
        map_func_3 = Function.from_dict(
            {"function": "map_func", "number": 3, "operator": "//"}
        )
        map_func_4 = Function.from_dict(
            {"function": "map_func", "number": 8, "operator": "//"}
        )

        # contraction
        for map_func_ in (map_func_2, map_func_3):
            composition = Composition(map_func_1)
            composition = Composition.from_composition(map_func_, composition)

            assert composition.root_function.operator is operator.mul
            assert (
                composition.root_function.number
                == map_func_1.number // map_func_.number
            )

        # no contraction
        composition = Composition(map_func_1)
        composition = Composition.from_composition(map_func_4, composition)

        assert composition.root_function.operator is operator.floordiv
        assert composition.root_function.number == map_func_4.number

    def test_add_sub_contraction(self):
        large_add = Function.from_dict(
            {"function": "map_func", "number": 6, "operator": "+"}
        )
        small_add = Function.from_dict(
            {"function": "map_func", "number": 3, "operator": "+"}
        )
        large_sub = Function.from_dict(
            {"function": "map_func", "number": 8, "operator": "-"}
        )
        small_sub = Function.from_dict(
            {"function": "map_func", "number": 2, "operator": "-"}
        )

        combinations = [
            (large_add, small_sub, operator.add, large_add.number - small_sub.number),
            (large_add, large_sub, operator.sub, large_sub.number - large_add.number),
            (small_sub, small_add, operator.add, small_add.number - small_sub.number),
            (large_sub, large_add, operator.sub, large_sub.number - large_add.number),
        ]

        for map_func_1, map_func_2, operator_, number in combinations:
            composition = Composition(map_func_1)
            composition = Composition.from_composition(map_func_2, composition)

            assert composition.root_function.operator is operator_
            assert composition.root_function.number == number

    def test_composition_merge(self):
        num_copies_in_comp = len(self.composition.copy_ids)
        num_copies_in_subcomp = len(self.composition2.copy_ids)
        sub_comps_to_add = [self.composition2, None, self.composition2]
        num_unique_subcomps = len(
            set(comp.id for comp in sub_comps_to_add if comp is not None)
        )

        copy_comp = copy.deepcopy(self.composition)

        copy_comp.extend_leaves(sub_comps_to_add, [None] * len(sub_comps_to_add))

        assert num_copies_in_comp + num_copies_in_subcomp + num_unique_subcomps == len(
            copy_comp.copy_ids
        )

        # visualize(copy_comp, "new_comp.png")


def main():
    tf = TestFunction()
    tf.test_from_dict()
    tf.test_eval_str_1_io_zip_with()
    tf.test_eval_str_1_io_other()
    tf.test_eval_more_io_zip_with()
    tf.test_eval_more_io_other()

    tc = TestComposition()
    tc.setup_method()
    tc.test_eval_str_leaves_1_io()
    tc.test_eval_more_io()
    tc.test_mul_floordiv_contractions()
    tc.test_add_sub_contraction()
    tc.test_composition_merge()


if __name__ == "__main__":
    main()
