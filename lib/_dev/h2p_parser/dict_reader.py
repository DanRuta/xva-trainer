# This reads a CMUDict formatted dictionary as a dictionary object
import re
import h2p_parser.format_ph as ph
from . import DATA_PATH


_dict_primary = 'cmudict.dict'


def read_dict(filename: str) -> list:
    # Read the file
    with open(filename, encoding='utf-8', mode='r') as f:
        # Read the file into lines
        lines = f.readlines()
    # Remove any line starting with ";;;"
    lines = [line for line in lines if not line.startswith(';;;')]
    return lines


def parse_dict(lines: list) -> dict:
    # Create a dictionary to store the parsed data
    parsed_dict = {}
    # Detect file format

    # We will read the first 10 lines to determine the format
    # Default to SSD format unless we find otherwise
    dict_form = 'SSD'
    for line in lines[:10]:
        # Strip new lines
        line = line.strip()
        if line == '':
            continue
        """
        Format 1 (Double Space Delimited):
        - Comment allowed to start with ";;;"
        WORD  W ER1 D
        
        Format 2 (Single Space Delimited):
        - Comment allowed at end of any line using "#"
        WORD W ER1 D # Comment
        """
        if '  ' in line:
            dict_form = 'DSD'
            break

    # Iterate over the lines
    for line in lines:
        # Skip empty lines and lines with no space
        line = line.strip()
        if line == '' and ' ' not in line:
            continue

        # Split depending on format
        if dict_form == 'DSD':
            pairs = line.split('  ')
        else:
            space_index = line.find(' ')
            line_split = line[:space_index], line[space_index + 1:]
            pairs = line_split[0], line_split[1].split('#')[0]

        word = str.lower(pairs[0])  # Get word and lowercase it
        phonemes = ph.to_list(pairs[1])   # Convert to list of phonemes
        phonemes = [phonemes]  # Wrap in nested list
        word_num = 0
        word_orig = None

        # Detect if this is a multi-word entry
        if ('(' in word) and (')' in word) and any(char.isdigit() for char in word):
            # Parse the integer from the word using regex
            result = int(re.findall(r"\((\d+)\)", word)[0])
            # If found
            if result is not None:
                # Set the original word
                word_orig = word
                # Remove the integer and bracket from the word
                word = re.sub(r"\(.*\)", "", word)
                # Set the word number to the result
                word_num = result

        # Check existing key
        if word in parsed_dict:
            # If word number is 0, ignore
            if word_num == 0:
                continue
            # If word number is not 0, add phoneme to existing key at index
            parsed_dict[word].extend(phonemes)
            # Also add the original word if it exists
            if word_orig is not None:
                parsed_dict[word_orig] = phonemes
        else:
            # Create a new key
            parsed_dict[word] = phonemes

    # Return the dictionary
    return parsed_dict


class DictReader:
    def __init__(self, filename=None):
        self.filename = filename
        self.dict = {}
        # If filename is None, use the default dictionary
        # default = 'data' uses the dictionary file in the data module
        # default = 'nltk' uses the nltk cmudict
        if filename is not None:
            self.dict = parse_dict(read_dict(filename))
        else:
            with DATA_PATH.joinpath(_dict_primary) as f:
                self.dict = parse_dict(read_dict(f))
