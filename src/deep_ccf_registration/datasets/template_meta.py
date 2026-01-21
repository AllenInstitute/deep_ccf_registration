from dataclasses import dataclass


@dataclass
class TemplateParameters:
    origin: tuple
    scale: tuple
    direction: tuple
    shape: tuple
    dims: int = 3
