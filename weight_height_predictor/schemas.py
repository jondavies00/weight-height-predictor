from dataclasses import dataclass


@dataclass
class ClassifierResults:
    errors: list[int]
    thetas: list[list[int]]