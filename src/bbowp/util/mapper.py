from bbowp.schema.bbowp import Choice


def map_variable_name(
    aliases: tuple[str, ...], params: dict, preprocess_name_fn=None
) -> tuple[str, ...]:
    """
    Map variable aliases to their corresponding parameter names.
    This function matches aliases against parameter names, optionally applying
    a preprocessing function to both for flexible name matching.
    Args:
        aliases: A tuple of variable names/aliases to search for.
        params: A dictionary of parameters whose keys will be matched against aliases.
        preprocess_name_fn: Optional function to preprocess both aliases and parameter
            names before comparison. If None, names are compared as-is. Defaults to None.
    Returns:
        A tuple of parameter names from `params` whose preprocessed names match
        any of the preprocessed aliases. Returns an empty tuple if no matches found.
    Example:
        >>> params = {'my_var': 1, 'my_var_2': 2, 'other': 3}
        >>> aliases = ('myVar',)
        >>> map_variable_name(aliases, params, preprocess_name_fn=lambda x: x.lower().replace('_', ''))
        ('my_var')
    """
    mapped_names = []

    aliases_preprocessed = tuple(
        preprocess_name_fn(alias) if preprocess_name_fn is not None else alias
        for alias in aliases
    )

    for name in params.keys():
        name_preprocessed = (
            preprocess_name_fn(name) if preprocess_name_fn is not None else name
        )
        if name_preprocessed in aliases_preprocessed:
            mapped_names.append(name)

    return tuple(mapped_names)


def map_choice_names(
    found_val: str, choices: tuple[Choice, ...], preprocess_name_fn=None
) -> tuple[str, ...]:
    """
    Map a found value to choice names by matching against choice aliases.
    Args:
        found_val: The value to search for, matched case-insensitively.
        choices: A tuple of Choice objects to search through.
    Returns:
        A tuple of actual choice names that match the found_val,
        either by their actual_name or any of their aliases.
    Example:
        If a Choice has actual_name='foo' and aliases=('f', 'fo'),
        calling map_choice_names('FO', (choice,)) returns ('foo',).
    """

    found_val_preprocessed = (
        preprocess_name_fn(found_val) if preprocess_name_fn is not None else found_val
    )
    mapped_choice_names = []

    for choice in choices:
        aliases = (choice.actual_name,) + choice.aliases
        aliases_preprocessed = tuple(
            preprocess_name_fn(alias) if preprocess_name_fn is not None else alias
            for alias in aliases
        )
        if found_val_preprocessed in aliases_preprocessed:
            mapped_choice_names.append(choice.actual_name)

    return tuple(mapped_choice_names)
