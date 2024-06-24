# -*- coding: utf-8 -*-
import itertools
import traceback
from typing import Container, Dict, Iterable, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from .base import BaseParser, ParseError


class TokenType(Enum):
    var = 1
    index_var = 2
    number = 3
    string = 4


@dataclass
class ParsedToken:
    value: str
    type: TokenType


class ConditionParser(BaseParser):
    """A class for parsing the condition string"""

    def __init__(
        self, line: str, allowed_vars: Union[Container[str], Iterable[str]] = []
    ):
        super(ConditionParser, self).__init__(line)
        self.parsed_code = ""
        self.parsed_decs: List[ParsedToken] = []
        self.allowed_vars = allowed_vars

    def parse(self):
        while not self._is_end():
            error_msg = self._read_next()
            if error_msg:
                return None, None, error_msg
        return self.parsed_code, self.parsed_decs, None

    @staticmethod
    def _is_keyword(w):
        return w == "and" or w == "or" or w == "in" or w == "not"

    @staticmethod
    def _is_operator(ch):
        return ch in ["=", "(", ")", "!", ">", "<"]

    def _throw(self, msg):
        msg = 'At character {} of "{}":\n\t{}'.format(self.i + 1, self.line, msg)
        raise ParseError(msg)

    def _maybe_read_index(self):
        # we only want to parse the LHS of ==
        if len(self.parsed_decs) % 2 == 1:
            return False, None

        if not self._is_end() and self._peek_char() == ".":
            # try to parse .index
            self._next_char()
            v = self._read_while(self._is_id)
            if v == "index":
                return (True, None)
            else:
                msg = 'Expected ".index", got ".{}"'.format(v)
                return False, msg

        return False, None

    def _read_next(self):
        self.parsed_code += self._read_while(BaseParser._is_whitespace)
        if self._is_end():
            return

        ch = self._peek_char()
        if self._is_id_start(ch):
            w = self._read_while(self._is_id)
            if ConditionParser._is_keyword(w):
                self.parsed_code += w
                return
            if w not in self.allowed_vars:
                msg = "*{}* is not a valid transform spec name. Allowed specs (those that have multiple branches): {}".format(
                    w, ", ".join([f"*{v}*" for v in self.allowed_vars])
                )
                return msg
            tk = ParsedToken(w, TokenType.var)
            is_ind, error_msg = self._maybe_read_index()
            if is_ind:
                tk.type = TokenType.index_var
            if error_msg:
                return error_msg

            self.parsed_decs.append(tk)
            self.parsed_code += "{}"
        elif self._is_string_start(ch):
            w = self._next_char() + self._read_while(lambda x: x != ch)
            if self._is_end() or self._peek_char() != ch:
                msg = 'Expected closing "{}"'.format(ch)
                return msg
            w += self._next_char()
            self.parsed_decs.append(ParsedToken(w, TokenType.string))
            self.parsed_code += "{}"
        elif self._is_digit(ch):
            w = self._read_while(self._is_digit)
            if not self._is_end() and self._peek_char() == ".":  # read decimal
                w += self._next_char() + self._read_while(self._is_digit)

            self.parsed_decs.append(ParsedToken(w, TokenType.number))
            self.parsed_code += "{}"
        elif self._is_operator(ch):
            w = self._read_while(ConditionParser._is_operator)
            self.parsed_code += w
        else:
            msg = 'Cannot handle character "{}"'.format(ch)
            msg = 'At character {} of "{}":\n{}'.format(self.i + 1, self.line, msg)
            return msg

    @staticmethod
    def make_index_var(w):
        """Embellish the variable name if the user is checking the option
        by its index in the options array."""
        return "_i_" + w

    def parse_and_recon(self):
        code, parsed_decs, error_msg = self.parse()
        if error_msg:
            return None, error_msg
        return self._recon(code, parsed_decs, self.line)

    @staticmethod
    def _recon(code: str, parsed_transforms: List[ParsedToken], cond: str):
        """Transform parsed code and decisions into valid python code"""
        exe = []
        for i, d in enumerate(parsed_transforms):
            if d.type == TokenType.index_var:
                exe.append(ConditionParser.make_index_var(d.value))
            elif d.type == TokenType.string:
                exe.append(d.value)

            elif (
                i % 2 == 1
                and d.type == TokenType.number
                and parsed_transforms[i - 1].type != TokenType.index_var
            ):  # the RHS of a number comparison should be an index var

                return (
                    None,
                    f"For `{parsed_transforms[i - 1].value} op {d.value}`, expecting an index to be on the LHS of a number comparison.",
                )
            else:
                exe.append(d.value)

        recon = code.format(*exe)

        # check if the code has syntax error
        error_msg = None
        try:
            eval(recon, {})
        except SyntaxError:
            msg = (
                "In parsing condition: "
                + f"'{cond}'"
                + "\nSyntax Error: invalid syntax"
            )
            msg += f"\n{traceback.format_exc()}"
            error_msg = msg
        except NameError:
            pass
        return recon, error_msg

    @staticmethod
    def eval_constraints_on_branch(
        condition: str,
        combination: Dict[str, Tuple[List[str], int]],
    ) -> bool:
        """Evaluate the condition on a combination of branch dependencies"""
        res = {}
        for spec_name, (deps, index) in combination.items():
            res[spec_name] = deps
            res[ConditionParser.make_index_var(spec_name)] = index
        try:
            return eval(condition, res), None
        except Exception as e:
            return False, str(e)


def parse_condition(line, allowed_vars):
    parser = ConditionParser(line, allowed_vars)
    code, parsed_decs, error_msg = parser.parse()
    if error_msg is None:
        code, error_msg = parser._recon(code, parsed_decs, line)
    return code, parsed_decs, error_msg


def get_conditions_from_branches(
    spec_name_to_branches_spec_ids: Dict[str, List[List[str]]]
) -> List[Dict[str, Tuple[List[str], int]]]:
    """Returns a List of dictionaries, each dictionary is a combination of branches
    Input:
     {
        "groupby": [
            ["derive_1", "derive_6"],
            ["derive_2", "derive_3"],
            ["derive_4", "derive_5"],
        ],
        "derive_7": [["f1"], ["derive_2", "f4"], ["derive_4", "derive_5"]],
    }

    E.g.
    [
        {'groupby': (['derive_1', 'derive_6'], 0), 'derive_7': (['f1'], 0)},
        {'groupby': (['derive_1', 'derive_6'], 0), 'derive_7': (['derive_2', 'f4'], 1)},
        {'groupby': (['derive_1', 'derive_6'], 0), 'derive_7': (['derive_4', 'derive_5'], 2)},
        ...
    ]

    """
    result = []

    for combination in itertools.product(*spec_name_to_branches_spec_ids.values()):
        res = {}
        for k, com_val in zip(spec_name_to_branches_spec_ids.keys(), combination):
            res[k] = (com_val, spec_name_to_branches_spec_ids[k].index(com_val))
        result.append(res)
    return result


def try_condition(condition, branches):
    allowed_vars = list(branches.keys())
    combinations = get_conditions_from_branches(branches)
    code, parsed_decs, error_msg = parse_condition(condition, allowed_vars)
    if error_msg:
        return error_msg
    ret = []
    for combination in combinations:
        res, msg = ConditionParser.eval_constraints_on_branch(code, combination)
        ret.append((combination, res, msg))
    return ret


if __name__ == "__main__":
    BRANCHES = {
        "groupby": [
            ["derive_1", "derive_6"],
            ["derive_2", "derive_3"],
            ["derive_4", "derive_5"],
        ],
        "derive_7": [["f1"], ["derive_2", "f4"], ["derive_4", "derive_5"]],
    }

    CONDITIONS = get_conditions_from_branches(BRANCHES)
    r1 = try_condition("groupby == 'derive_1' and derive_7 == 'f1'", BRANCHES)
    r2 = try_condition("groupby == 'derive_1' and derive_7 == 'f4'", BRANCHES)
    r3 = try_condition("'derive_1'in groupby or derive_7.index == 2", BRANCHES)

    # c1, p1, e1 = parse_condition("'asdas' in a_b and b and cindex")
    # c2, p2, e2 = parse_condition("and >  b.index > 3.5 and c.index == 4")
    # c3, p3, e3 = parse_condition("a.index > 2 and b.index < 3.5 and c.index == 4")
    print("here")
