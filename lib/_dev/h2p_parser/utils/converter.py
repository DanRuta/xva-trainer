# Converts dictionary files
import json
import os

from .. import symbols
from .. import format_ph as ph
from tqdm import tqdm


def from_binary_delim(path, delimiter) -> dict:
    # Converts a delimited binary state heteronym look-up dictionary to a dict format
    # Expected format: WORD|(Space Seperated Phonemes Case)|(Space Seperated Phonemes Default)|(Case)
    # Example: "REJECT|R IH0 JH EH1 K T|R IY1 JH EH0 K T|V"
    # Hashtag comments are allowed but only at the start of a file

    # Import file
    result_dict = {}
    num_lines = sum(1 for line in open(path, 'r'))
    with open(path, 'r') as f:
        skipped_comments = False
        for line in tqdm(f, total=num_lines):
            # Skip comments
            if not skipped_comments:
                if line.startswith('#') or line == '\n':
                    continue
                else:
                    skipped_comments = True
            # Skip empty or newline lines
            if line.strip() == '' or line.strip() == '\n':
                continue
            # Parse line using passed delimiter
            tokens = line.strip().split(delimiter)
            # Check for correct number of tokens
            if len(tokens) != 4:
                raise ValueError('Invalid number of tokens in line: ' + line)
            # Get word (token 0) and check validity (no spaces)
            word = tokens[0].lower()
            if ' ' in word:
                raise ValueError('Invalid word in line: ' + line)
            # Get phonemes and check validity (alphanumeric)
            ph_case = tokens[1]
            ph_default = tokens[2]
            if not ph_case.replace(' ', '').isalnum() or not ph_default.replace(' ', '').isalnum():
                raise ValueError('Invalid phonemes in line: ' + line)
            # Get case (token 3) and check validity (alphanumeric)
            case = tokens[3]
            if not case.isalnum():
                raise ValueError('Invalid case in line: ' + line)
            # Check if case is a full case or full type case
            if case in symbols.pos_tags_set or case in symbols.pos_type_tags_set:
                # Add to dictionary directly
                # Build sub-dictionary for each case
                sub_dict = result_dict.get(word, {})
                sub_dict[case] = ph.to_sds(ph_case)
                sub_dict['DEFAULT'] = ph.to_sds(ph_default)
                result_dict[word] = sub_dict
            # Check if case is a short type case
            elif case in symbols.pos_type_short_tags_set:
                # Need to convert to full type case
                sub_dict = result_dict.get(word, {})
                case_short = symbols.pos_type_form_dict[case]
                sub_dict[case_short] = ph.to_sds(ph_case)
                sub_dict['DEFAULT'] = ph.to_sds(ph_default)
                result_dict[word] = sub_dict
            else:
                raise ValueError('Invalid case in line: ' + line)
    return result_dict


# Method to write a dict to a json file
def to_json(path, dict_to_write):
    # Writes a dictionary to a json file
    with open(path, 'w') as f:
        json.dump(dict_to_write, f, indent=4, sort_keys=True)


# Combined method to convert binary delimited files to json
def bin_delim_to_json(path, output_path, delimiter):
    to_json(output_path, from_binary_delim(path, delimiter))
