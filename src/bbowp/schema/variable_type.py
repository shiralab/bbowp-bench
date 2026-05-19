from enum import StrEnum


class VariableType(StrEnum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    CATEGORICAL = "cat"


class ConstVariableType(StrEnum):
    INT = "const_int"
    FLOAT = "const_float"
    BOOL = "const_bool"
    CATEGORICAL = "const_cat"
