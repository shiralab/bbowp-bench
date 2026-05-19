import re
from collections.abc import Sequence


def convert_snake_case(
    text: str,
    delimiters: Sequence[str] = (" ", "_", "."),
    stop_words: Sequence[str] = ("of", "for", "at", "by", "a", "an", "the"),
):
    text = text.lower()  # lower characters
    terms = re.split(
        "[{}]".format("".join(delimiters)), text
    )  # split to terms by delimiters
    terms = [
        t for t in terms if t not in stop_words and t != ""
    ]  # remove stop words that are less informative and can be noise, and remove empty terms
    return "_".join(terms)
