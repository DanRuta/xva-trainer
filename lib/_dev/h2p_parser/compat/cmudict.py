# Compatibility layer for using CMUDictExt with CMUDict-like API calls.
# Designed to be compatible with the implementation of CMUDict in:
# https://github.com/NVIDIA/DeepLearningExamples/
#
# Example usage:
#   from h2p_parser.compat.cmudict import CMUDict

from h2p_parser.cmudictext import CMUDictExt


class CMUDict(CMUDictExt):
    def __init__(self, file_or_path=None, heteronyms_path=None, keep_ambiguous=True):
        # Parameter Mapping:
        # file_or_path => Mapped to cmu_dict_path
        # heteronyms_path => Dropped as CMUDictExt uses H2p for heteronym parsing.
        # keep_ambiguous => Mapped to cmu_multi_mode | True => -2, False => -1
        super().__init__(file_or_path, heteronyms_path)
        self._entries = {}
        self.heteronyms = []
