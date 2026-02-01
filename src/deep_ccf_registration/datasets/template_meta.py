from dataclasses import dataclass


@dataclass
class TemplateParameters:
    origin: tuple
    scale: tuple
    direction: tuple
    shape: tuple
    orientation: str
    dims: int = 3
