from unicodedata import normalize
import re

# Pre-compile regex
re_filter = re.compile(r"[^ A-Za-z'.,?!()\-]")
re_filter_with_num = re.compile(r"[^ A-Za-z\d'.,?!()\-]")
re_multi_space = re.compile(r"\s\s+")


# Filters text before parsing
# @param text: text to be filtered
# @return: filtered text
def filter_text(text: str, allow_num: bool = False, preserve_case: bool = False) -> str:
    """
    Filters text before parsing
    :param preserve_case:
    :param allow_num: True if numbers are allowed
    :param text: Input raw text
    :return: Text after stripped accents, lower-cased, and invalid punctuation removed
    """
    # Strip accents
    text = normalize('NFD', text)
    # To lowercase
    if not preserve_case:
        text = text.lower()
    # Remove all invalid punctuation
    if allow_num:
        text = re.sub(re_filter_with_num, '', text)
    else:
        text = re.sub(re_filter, "", text)
    # Remove all spaces more than 1
    text = re.sub(re_multi_space, " ", text)
    # Return
    return text
