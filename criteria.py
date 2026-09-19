import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CriteriaToken:
    token_type: str
    value: str


class Condition(ABC):
    @abstractmethod
    def evaluate(self, scores_by_identifier: dict[str, float], threshold: float) -> bool:
        pass

    @abstractmethod
    def calculate_score(self, scores_by_identifier: dict[str, float]) -> float:
        pass

    @abstractmethod
    def collect_criteria(self) -> list["Criterion"]:
        pass


class Criterion(Condition):
    def __init__(self, identifier: str, instruction: str):
        self.identifier = identifier
        self.instruction = instruction

    def evaluate(self, scores_by_identifier: dict[str, float], threshold: float) -> bool:
        criterion_score = scores_by_identifier.get(self.identifier, 0.0)
        return criterion_score >= threshold

    def calculate_score(self, scores_by_identifier: dict[str, float]) -> float:
        return scores_by_identifier.get(self.identifier, 0.0)

    def collect_criteria(self) -> list["Criterion"]:
        return [self]


class AndCondition(Condition):
    def __init__(self, *conditions: Condition):
        self.conditions = list(conditions)

    def add_condition(self, condition: Condition) -> None:
        self.conditions.append(condition)

    def evaluate(self, scores_by_identifier: dict[str, float], threshold: float) -> bool:
        if not self.conditions:
            return True
        return all(condition.evaluate(scores_by_identifier, threshold) for condition in self.conditions)

    def calculate_score(self, scores_by_identifier: dict[str, float]) -> float:
        if not self.conditions:
            return 0.0
        scores = [condition.calculate_score(scores_by_identifier) for condition in self.conditions]
        return min(scores)

    def collect_criteria(self) -> list[Criterion]:
        collected_criteria = []
        for condition in self.conditions:
            collected_criteria.extend(condition.collect_criteria())
        return collected_criteria


class OrCondition(Condition):
    def __init__(self, *conditions: Condition):
        self.conditions = list(conditions)

    def add_condition(self, condition: Condition) -> None:
        self.conditions.append(condition)

    def evaluate(self, scores_by_identifier: dict[str, float], threshold: float) -> bool:
        if not self.conditions:
            return False
        return any(condition.evaluate(scores_by_identifier, threshold) for condition in self.conditions)

    def calculate_score(self, scores_by_identifier: dict[str, float]) -> float:
        if not self.conditions:
            return 0.0
        scores = [condition.calculate_score(scores_by_identifier) for condition in self.conditions]
        return max(scores)

    def collect_criteria(self) -> list[Criterion]:
        collected_criteria = []
        for condition in self.conditions:
            collected_criteria.extend(condition.collect_criteria())
        return collected_criteria


class CriteriaTokenizer:
    TOKEN_PATTERN = re.compile(
        r'(\(|\)|\"[^\"]*\"|\'[^\']*\'|\bAND\b|\bOR\b|[^\s()\"\']+)',
        re.IGNORECASE,
    )

    def tokenize(self, expression: str) -> list[CriteriaToken]:
        # Split criteria expression into semantic tokens including parentheses and quotes.
        tokens: list[CriteriaToken] = []
        pending_words: list[str] = []

        def flush_pending_words() -> None:
            if pending_words:
                combined_text = " ".join(pending_words).strip()
                if combined_text:
                    tokens.append(CriteriaToken(token_type="TEXT", value=combined_text))
                pending_words.clear()

        for match in self.TOKEN_PATTERN.finditer(expression):
            token_str = match.group()
            token_upper = token_str.upper()

            if token_upper in ("AND", "OR", "(", ")"):
                flush_pending_words()
                if token_upper == "(":
                    tokens.append(CriteriaToken(token_type="LPAREN", value="("))
                elif token_upper == ")":
                    tokens.append(CriteriaToken(token_type="RPAREN", value=")"))
                else:
                    tokens.append(CriteriaToken(token_type=token_upper, value=token_upper))
            elif (token_str.startswith('"') and token_str.endswith('"')) or (
                token_str.startswith("'") and token_str.endswith("'")
            ):
                flush_pending_words()
                tokens.append(CriteriaToken(token_type="TEXT", value=token_str[1:-1]))
            else:
                pending_words.append(token_str)

        flush_pending_words()
        return tokens


class CriteriaExpressionParser:
    def __init__(self, tokenizer: CriteriaTokenizer | None = None):
        self.tokenizer = tokenizer or CriteriaTokenizer()
        self._criterion_counter = 0
        self._tokens: list[CriteriaToken] = []
        self._current_index = 0

    def _generate_identifier(self) -> str:
        self._criterion_counter += 1
        return f"criterion_{self._criterion_counter}"

    def _peek_token(self) -> CriteriaToken | None:
        if self._current_index < len(self._tokens):
            return self._tokens[self._current_index]
        return None

    def _consume_token(self, expected_type: str | None = None) -> CriteriaToken | None:
        token = self._peek_token()
        if token is not None and (expected_type is None or token.token_type == expected_type):
            self._current_index += 1
            return token
        return None

    def _parse_factor(self) -> Condition:
        token = self._peek_token()
        if token is not None and token.token_type == "LPAREN":
            self._consume_token("LPAREN")
            nested_condition = self._parse_or()
            self._consume_token("RPAREN")
            return nested_condition

        if token is not None and token.token_type == "TEXT":
            self._consume_token("TEXT")
            return Criterion(identifier=self._generate_identifier(), instruction=token.value)

        return Criterion(identifier=self._generate_identifier(), instruction="")

    def _parse_and(self) -> Condition:
        left_node = self._parse_factor()
        and_conditions: list[Condition] = [left_node]

        while True:
            peeked = self._peek_token()
            if peeked is not None and peeked.token_type == "AND":
                self._consume_token("AND")
                and_conditions.append(self._parse_factor())
            else:
                break

        if len(and_conditions) == 1:
            return and_conditions[0]
        return AndCondition(*and_conditions)

    def _parse_or(self) -> Condition:
        left_node = self._parse_and()
        or_conditions: list[Condition] = [left_node]

        while True:
            peeked = self._peek_token()
            if peeked is not None and peeked.token_type == "OR":
                self._consume_token("OR")
                or_conditions.append(self._parse_and())
            else:
                break

        if len(or_conditions) == 1:
            return or_conditions[0]
        return OrCondition(*or_conditions)

    def parse(self, expression: str) -> Condition:
        # Parse boolean criteria string with operator precedence and parentheses into AST.
        trimmed = expression.strip()
        if not trimmed:
            return Criterion(identifier=self._generate_identifier(), instruction="")

        self._tokens = self.tokenizer.tokenize(trimmed)
        self._current_index = 0
        return self._parse_or()
