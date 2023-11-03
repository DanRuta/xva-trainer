# Parses annotation files for conversion of sentences to phonemes
from __future__ import annotations
from h2p_parser import cmudictext
from h2p_parser.filter import filter_text
from h2p_parser.text.numbers import normalize_numbers
from h2p_parser.symbols import punctuation

# Reads a file into a list of lines
from tqdm import tqdm


def read_file(file_name, delimiter) -> list:
    with open(file_name, 'r', encoding="utf-8") as f:
        result = []
        for line in f:
            line = line.split(delimiter)
            # Take the second element
            result.append(line[1].lower())
        return result

# Method that checks if a single line is resolvable


# Checks a list of lines for unresolvable words
# Returns a list of lines with unresolvable words, or None if no unresolvable words
def check_lines(lines: list) -> ParseResult:
    cde = cmudictext.CMUDictExt()
    # Holds result
    result = ParseResult()
    # Loop with nqdm
    for line in tqdm(lines, desc='Checking lines'):
        # Add
        result.all_lines.append(line)
        result.lines.add(line)
        # If line contains het, add to result
        if cde.h2p.contains_het(line):
            result.all_lines_cont_het.append(line)
        # Filter the line
        f_line = filter_text(line)
        # Number converter
        f_line = normalize_numbers(f_line)
        # Tokenize
        tokens = cde.h2p.tokenize(f_line)
        for word in tokens:
            # Skip word if punctuation
            if word in punctuation:
                continue
            # Add word to result
            result.all_words.append(word)
            result.words.add(word)
            # Check if word is resolvable
            h2p_res = cde.h2p.contains_het(word)
            cmu_res = cde.dict.get(word) is not None
            fet_res = cde.lookup(word) is not None
            if not h2p_res and not cmu_res and not fet_res:
                # If word ends in "'s", remove it and add the base word
                if word.endswith("'s"):
                    word = word[:-2]
                result.unres_all_lines.append(line)
                result.unres_all_words.append(word)
                result.unres_lines.add(line)
                result.unres_words.add(word)
            elif h2p_res:
                result.n_words_res += 1
                result.n_words_het += 1
            elif cmu_res:
                result.n_words_res += 1
                result.n_words_cmu += 1
            elif fet_res:
                result.n_words_res += 1
                result.n_words_fet += 1

    # Also pass stats
    result.ft_stats = cde.p.stat_resolves

    return result


# Class to hold the result of a parse
class ParseResult:
    def __init__(self):
        self.all_lines = []
        self.all_lines_cont_het = []
        self.unres_all_lines = []
        self.lines = set()
        self.unres_lines = set()
        # Words
        self.all_words = []
        self.unres_all_words = []
        self.words = set()
        self.unres_words = set()
        # Numerical stats
        self.n_words_res = 0  # Number of total resolved words
        self.n_words_cmu = 0  # Resolved words from CMU
        self.n_words_fet = 0  # Resolved words from Features
        self.n_words_het = 0  # Resolved words from H2p
        # Stats from cmudictext
        self.ft_stats = None

    # Get percentage of lines covered
    def line_unique_coverage(self) -> float:
        dec = 1 - len(self.unres_lines) / len(self.lines)
        return round(dec * 100, 2)

    # Get percentage of words covered
    def word_unique_coverage(self) -> float:
        dec = 1 - len(self.unres_words) / len(self.words)
        return round(dec * 100, 2)

    # Get percentage of lines covered (All)
    def line_coverage(self) -> float:
        dec = 1 - len(self.unres_all_lines) / len(self.all_lines)
        return round(dec * 100, 2)

    # Get percentage of words covered (All)
    def word_coverage(self) -> float:
        dec = 1 - len(self.unres_all_words) / len(self.all_words)
        return round(dec * 100, 2)

    # Get percentage of heteronyms containing lines
    def percent_line_het(self) -> float:
        dec = len(self.all_lines_cont_het) / len(self.all_lines)
        return round(dec * 100, 2)

    # Get percentage of words resolved by H2p
    def percent_word_h2p(self) -> float:
        dec = self.n_words_het / self.n_words_res
        return round(dec * 100, 2)

    # Get percentage of words resolved by CMU
    def percent_word_cmu(self) -> float:
        dec = self.n_words_cmu / self.n_words_res
        return round(dec * 100, 2)
