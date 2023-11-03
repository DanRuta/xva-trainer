from collections import Counter

from InquirerPy import inquirer
from InquirerPy.utils import patched_print, color_print
from InquirerPy.base.control import Choice
from InquirerPy.validator import PathValidator
from h2p_parser.utils import converter
from h2p_parser.utils import parser


def convert_h2p(input_file, output_file, delimiter):
    """
    Converts a h2p dictionary file from one format to another.
    """
    converter.bin_delim_to_json(input_file, output_file, delimiter)
    print('Converted h2p_dict to json.')


def prompt_action() -> str:
    action = inquirer.select(
        message='Select action:',
        choices=[
            "Convert",
            "Parse",
            Choice(value=None, name='Exit')
        ],
        default=0,
    ).execute()
    if not action:
        exit(0)
    return action


def prompt_f_input():
    """
    Prompts for input file.
    """
    return inquirer.filepath(
        message='Select input file:',
        validate=PathValidator(is_file=True, message='Input must be a file.')
    ).execute()


def prompt_f_output():
    """
    Prompts for output file.
    """
    return inquirer.filepath(
        message='Select output file:',
        validate=PathValidator(is_file=True, message='Output must be a file.')
    ).execute()


def action_convert():
    """
    Converts a h2p dictionary file from one format to another.
    """
    # Select input file
    input_file = prompt_f_input()
    if not input_file:
        return

    # Select output file
    output_file = prompt_f_output()
    if not output_file:
        return

    # Ask for delimiter
    delimiter = inquirer.text(
        message='Enter delimiter:',
        default='|'
    ).execute()
    if not delimiter:
        return

    # Run Process
    convert_h2p(input_file, output_file, delimiter)


def action_parse_file():
    """
    Parses a metadata.csv file and checks for dictionary coverage
    :return:
    """
    # Select input file
    input_file = prompt_f_input()
    if not input_file:
        return

    # Ask for delimiter
    delimiter = inquirer.text(
        message='Enter delimiter:',
        default='|'
    ).execute()
    if not delimiter:
        return

    # Run Process
    result = parser.check_lines(parser.read_file(input_file, delimiter))

    # Print results
    color_print([("#e5c07b", "Unresolved Words")])
    color_print([("#d21205", "[All]: "),
                 ("#ffffff", f"{len(result.unres_all_words)}/{len(result.all_words)}")])
    color_print([("#7e3b41", "[Unique]: "),
                 ("#ffffff", f"{len(result.unres_words)}/{len(result.words)}")])

    color_print([("#4ce5c8", "-" * 10)])

    color_print([("#e5c07b", "Unresolved Lines")])
    color_print([("#d21205", "[All]: "),
                 ("#ffffff", f"{len(result.unres_all_lines)}/{len(result.all_lines)}")])
    color_print([("#7e3b41", "[Unique]: "),
                 ("#ffffff", f"{len(result.unres_lines)}/{len(result.lines)}")])

    color_print([("#4ce5c8", "-" * 10)])

    color_print([("#e5c07b", "Expected Coverage")])
    color_print([("#d21205", "[Lines]: "),
                 ("#ffffff", f"{result.line_coverage()}%")])
    color_print([("#7e3b41", "[Words]: "),
                 ("#ffffff", f"{result.word_coverage()}%")])

    color_print([("#4ce5c8", "-" * 10)])

    color_print([("#e5c07b", "H2p parser")])
    color_print([("#d21205", "[Lines with Heteronyms]: "),
                 ("#ffffff", f"{len(result.all_lines_cont_het)}/{len(result.all_lines)}"
                             f" | {result.percent_line_het()}%")])
    color_print([("#7e3b41", "[Words Resolved by H2p]: "),
                 ("#ffffff", f"{result.n_words_het}/{result.n_words_res}"
                             f" | {result.percent_word_h2p()}%")])
    # Calcs
    feature_res = result.n_words_fet
    feature_percent = round(feature_res / result.n_words_res * 100, 2)
    cmu_res = result.n_words_cmu
    cmu_percent = round(cmu_res / result.n_words_res * 100, 2)
    color_print([("#c8bd20", "[Transformed Resolves]: "),
                 ("#ffffff", f"{feature_res}/{result.n_words_res}"
                             f" | {feature_percent}%")])
    color_print([("#25a0c8", "[Words in CMUDict]: "),
                 ("#ffffff", f"{cmu_res}/{result.n_words_res}"
                             f" | {cmu_percent}%")])

    color_print([("#4ce5c8", "-" * 10)])

    color_print([("#e5c07b", "Feature Usage")])

    # Loop through feature results
    for ft in result.ft_stats:
        color_print([("#d21205", f"{ft}: "),
                     ("#ffffff", f"{result.ft_stats[ft]}/{result.n_words_res}"
                                 f" | {round(result.ft_stats[ft]/result.n_words_res*100, 2)}%")])

    color_print([("#4ce5c8", "-" * 10)])

    # Print 100 sampled unresolved words by frequency
    color_print([("#e5c07b", "Top 100 most frequent unresolved words")])
    # Count frequency of words
    word_freq = Counter(result.unres_all_words)
    # Sort by frequency
    word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    # Print top 100
    for word, freq in word_freq[:100]:
        color_print([("#d21205", f"{word}: "),
                     ("#ffffff", f"{freq}")])


def entry():
    """
    Prints help information.
    """
    # Select action type
    action = prompt_action()
    if action == 'Convert':
        action_convert()
    elif action == 'Parse':
        action_parse_file()


if __name__ == "__main__":
    entry()



