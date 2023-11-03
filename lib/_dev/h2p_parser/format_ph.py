from typing import overload

# Converts and outputs various formats of phonemes


@overload
def to_sds(ph: str) -> str: ...


@overload
def to_sds(ph: list) -> str: ...


def to_sds(ph: list or str) -> str or None:
    """
    Converts phonemes to space delimited string format

    :param ph: Phoneme as str or list, supports nested lists
    :return: Phoneme as space delimited string
    """
    # Return None if None
    if ph is None:
        return None

    # Return directly if str
    if isinstance(ph, str):
        return ph
    # If is list, convert each element
    if isinstance(ph, list):
        # If list empty, return None
        if len(ph) == 0:
            return None
        # Case for further lists
        if isinstance(ph[0], list):
            return to_sds(ph[0])  # Recursive call
        # Case if str at index 0, and size 1, return directly
        elif isinstance(ph[0], str) and len(ph) == 1:
            return ph[0]
        # Case if str at index 0, above size 1, return with join
        elif isinstance(ph[0], str):
            return ' '.join(ph)
        # Case for none
        elif ph[0] is None:
            return None
        else:
            raise TypeError('to_sds() encountered an unexpected nested element type')
    # Error if no matches
    raise TypeError('to_sds() expects a list or string')


@overload
def to_list(ph: str) -> list: ...


@overload
def to_list(ph: list) -> list: ...


def to_list(ph: str or list) -> list or None:
    """
    Converts phonemes to list format

    :param ph: Phoneme as str or list, supports nested lists
    :return: Phoneme as list
    """
    # Return None if None
    if ph is None:
        return None

    # Return directly if list and index 0 is str
    if isinstance(ph, list) and len(ph) > 0 and isinstance(ph[0], str):
        return ph

    # If space delimited string, convert to list
    if isinstance(ph, str):
        return ph.split(' ')

    # If nested list, convert each element
    if isinstance(ph, list):
        # If list empty or has None, return None
        if len(ph) == 0 or ph[0] is None:
            return None
        # Case for further lists
        if isinstance(ph[0], list):
            return to_list(ph[0])  # Recursive call

    # Error if no matches
    raise TypeError('to_list() expects a list or string')


# Surrounds text with curly brackets
def with_cb(text: str) -> str:
    """
    Surrounds text with curly brackets

    :param text: Text to surround
    :return: Surrounded text
    """
    return '{' + text + '}'
