# Transformations of text sequences for matching
from __future__ import annotations
from typing import TYPE_CHECKING
from .symbols import consonants

import re

if TYPE_CHECKING:
    from .cmudictext import CMUDictExt

_re_digit = re.compile(r'\d+')


class Processor:
    def __init__(self, cde: CMUDictExt):
        self._lookup = cde.lookup
        self._cmu_get = cde.dict.get
        self._segment = cde.segment
        self._tag = cde.h2p.tag
        self._stem = cde.stem
        # Number of times respective methods were called
        self.stat_hits = {
            'plural': 0,
            'possessives': 0,
            'contractions': 0,
            'hyphenated': 0,
            'compound': 0,
            'compound_l2': 0,
            'stem': 0
        }
        # Number of times respective methods returned value (not None)
        self.stat_resolves = {
            'plural': 0,
            'possessives': 0,
            'contractions': 0,
            'hyphenated': 0,
            'compound': 0,
            'compound_l2': 0,
            'stem': 0
        }
        # Holds events when features encountered unexpected language syntax
        self.stat_unexpected = {
            'plural': [],
            'possessives': [],
            'contractions': [],
            'hyphenated': [],
            'compound': [],
            'compound_l2': [],
            'stem': []
        }

    def auto_possessives(self, word: str) -> str | None:
        """
        Auto-possessives
        :param word: Input of possible possessive word
        :return: Phoneme of word as SDS, or None if unresolvable
        """
        if not word.endswith("'s"):
            return None
        # If the word ends with "'s", register a hit
        self.stat_hits['possessives'] += 1
        """
        There are 3 general cases:
        1. Base words ending in one of 6 special consonants (sibilants)
            - i.e. Tess's, Rose's, Butch's, Midge's, Rush's, Garage's
            - With consonants ending of [s], [z], [ch], [j], [sh], [zh]
            - In ARPAbet: {S}, {Z}, {CH}, {JH}, {SH}, {ZH}
            - These require a suffix of {IH0 Z}
        2. Base words ending in vowels and voiced consonants:
            - i.e. Fay's, Hugh's, Bob's, Ted's, Meg's, Sam's, Dean's, Claire's, Paul's, Bing's
            - In ARPAbet: {IY0}, {EY1}, {UW1}, {B}, {D}, {G}, {M}, {N}, {R}, {L}, {NG}
            - Vowels need a wildcard match of any numbered variant
            - These require a suffix of {Z}
        3. Base words ending in voiceless consonants:
            - i.e. Hope's, Pat's, Clark's, Ruth's
            - In ARPAbet: {P}, {T}, {K}, {TH}
            - These require a suffix of {S}
        """

        # Method to return phoneme and increment stat
        def _resolve(phoneme: str) -> str:
            self.stat_resolves['possessives'] += 1
            return phoneme

        core = word[:-2]  # Get core word without possessive
        ph = self._lookup(core, ph_format='list')  # find core word using recursive search
        if ph is None:
            return None  # Core word not found
        # [Case 1]
        if ph[-1] in {'S', 'Z', 'CH', 'JH', 'SH', 'ZH'}:
            ph += 'IH0' + 'Z'
            return _resolve(ph)
        # [Case 2]
        """
        Valid for case 2:
        'AA', 'AO', 'EY', 'OW', 'UW', 'AE', 'AW', 'EH', 'IH', 
        'OY', 'AH', 'AY', 'ER', 'IY', 'UH', 'UH', 
        'B', 'D', 'G', 'M', 'N', 'R', 'L', 'NG'
        To simplify matching, we will check for the listed single-letter variants and 'NG'
        and then check for any numbered variant
        """
        if ph[-1] in {'B', 'D', 'G', 'M', 'N', 'R', 'L', 'NG'} or ph[-1][-1].isdigit():
            ph += 'Z'
            return _resolve(ph)
        # [Case 3]
        if ph[-1] in ['P', 'T', 'K', 'TH']:
            ph += 'S'
            return _resolve(ph)

        return None  # No match found

    def auto_contractions(self, word: str) -> str | None:
        """
        Auto contracts form and finds phonemes
        :param word:
        :return:
        """
        """
        Supported contractions:
        - 'll
        - 'd
        """
        # First, check if the word is a contraction
        parts = word.split("\'")  # Split on [']
        if len(parts) == 1 or parts[1] not in {'ll', 'd'}:
            return None  # No contraction found
        if len(parts) > 2:
            self.stat_unexpected['contraction'] += word
            return None  # More than 2 parts, can't be a contraction
        # If initial check passes, register a hit
        self.stat_hits['contractions'] += 1

        # Get the core word
        core = parts[0]
        # Get the phoneme for the core word recursively
        ph = self._lookup(core, ph_format='list')
        if ph is None:
            return None  # Core word not found
        # Add the phoneme with the appropriate suffix
        if parts[1] == 'll':
            ph += 'L'
        elif parts[1] == 'd':
            ph += 'D'
        # Return the phoneme
        self.stat_resolves['contractions'] += 1
        return ph

    def auto_hyphenated(self, word: str) -> str | None:
        """
        Splits hyphenated words and attempts to resolve components
        :param word:
        :return:
        """
        # First, check if the word is a hyphenated word
        if '-' not in word:
            return None  # No hyphen found
        # If initial check passes, register a hit
        self.stat_hits['hyphenated'] += 1
        # Split the word into parts
        parts = word.split('-')
        # Get the phonemes for each part
        ph = []
        for part in parts:
            ph_part = self._lookup(part, ph_format='sds')
            if ph_part is None:
                return None  # Part not found
            ph.append(ph_part)
        # Join the phonemes
        ph = ' '.join(ph)
        # Return the phoneme
        self.stat_resolves['hyphenated'] += 1
        return ph

    def auto_compound(self, word: str) -> str | None:
        """
        Splits compound words and attempts to resolve components
        :param word:
        :return:
        """
        # Split word into parts
        parts = self._segment(word)
        if len(parts) == 1:
            return None  # No compound found
        # If initial check passes, register a hit
        self.stat_hits['compound'] += 1
        # Get the phonemes for each part
        ph = []
        for part in parts:
            ph_part = self._lookup(part, ph_format='sds')
            if ph_part is None:
                return None  # Part not found
            ph.append(ph_part)
        # Join the phonemes
        ph = ' '.join(ph)
        # Return the phoneme
        self.stat_resolves['compound'] += 1
        return ph

    def auto_plural(self, word: str, pos: str = None) -> str | None:
        """
        Finds singular form of plurals and attempts to resolve separately
        Optionally a pos tag can be provided.
        If no tags are provided, there will be a single word pos inference,
        which is not ideal.
        :param pos:
        :param word:
        :return:
        """
        # First, check if the word is a replaceable plural
        # Needs to end in 's' or 'es'
        if word[-1] != 's':
            return None  # No plural found
        # Now check if the word is a plural using pos
        if pos is None:
            pos = self._tag(word)
        if pos is None or len(pos) == 0 or (pos[0] != 'NNS' and pos[0] != 'NNPS'):
            return None  # No tag found
        # If initial check passes, register a hit
        self.stat_hits['plural'] += 1

        """
        Case 1:
        > Word ends in 'oes'
        > Remove the 'es' to get the singular
        """
        if len(word) > 3 and word[-3:] == 'oes':
            singular = word[:-2]
            # Look up the possessive form (since the pronunciation is the same)
            ph = self.auto_possessives(singular + "'s")
            if ph is not None:
                self.stat_resolves['plural'] += 1
                return ph  # Return the phoneme

        """
        Case 2:
        > Word ends in 's'
        > Remove the 's' to get the singular
        """
        if len(word) > 1 and word[-1] == 's':
            singular = word[:-1]
            # Look up the possessive form (since the pronunciation is the same)
            ph = self.auto_possessives(singular + "'s")
            if ph is not None:
                self.stat_resolves['plural'] += 1
                return ph  # Return the phoneme

        # If no matches, return None
        return None

    def auto_stem(self, word: str) -> str | None:
        """
        Attempts to resolve using the root stem of a word.
        Supported modes:
            - "ing"
            - "ingly"
            - "ly"
        :param word:
        :return:
        """

        # noinspection SpellCheckingInspection
        """
        'ly' has no special rules, always add phoneme 'L IY0'
        
        'ing' relevant rules:
        
        > If the original verb ended in [e], remove it and add [ing]
            - i.e. take -> taking, make -> making
            - We will search once with the original verb, and once with [e] added
                - 1st attempt: tak, mak
                - 2nd attempt: take, make
            
        > If the input word has a repeated consonant before [ing], it's likely that
        the original verb has only 1 of the consonants
            - i.e. running -> run, stopping -> stop
            - We will search for repeated consonants, and perform 2 attempts:
                - 1st attempt: without the repeated consonant (run, stop)
                - 2nd attempt: with the repeated consonant (runn, stopp)
        """
        # Discontinue if word is too short
        if len(word) < 3 or (not word.endswith('ly') and not word.endswith('ing')):
            return None
        # Register a hit
        self.stat_hits['stem'] += 1  # Register hit

        # For ly case
        if word.endswith('ly'):
            # Get the root word
            root = word[:-2]
            # Recursively get the root
            ph_root = self._lookup(root, ph_format='sds')
            # If not exist, return None
            if ph_root is None:
                return None
            ph_ly = 'L IY0'
            ph_joined = ' '.join([ph_root, ph_ly])
            self.stat_resolves['stem'] += 1
            return ph_joined

        # For ing case
        if word.endswith('ing'):
            # Get the root word
            root = word[:-3]
            # Recursively get the root
            ph_root = self._lookup(root, ph_format='sds')
            # If not exist, return None
            if ph_root is None:
                return None
            ph_ly = 'IH0 NG'
            ph_joined = ' '.join([ph_root, ph_ly])
            self.stat_resolves['stem'] += 1
            return ph_joined

    def auto_component(self, word: str) -> str | None:
        """
        Searches for target word as component of a larger word
        :param word:
        :return:
        """

        """
        This processing step checks for words as a component of a larger word
        - i.e. 'synth' is not in the cmu dictionary
        - Stage 1: We will search for any word beginning with 'synth' (10 matches)
            - This is because most unseen short words are likely shortened versions
            - We will split 
        - Stage 2: Search for any word containing 'synth' (13 matches)
        
        """
        raise NotImplementedError

    def auto_compound_l2(self, word: str, recursive: bool = True) -> str | None:
        """
        Searches for target word as a compound word.
        > Does not use n-gram splitting like auto_compound()
        > Splits words manually into every possible combination
        > Returns the match with the highest length of both words
        :param recursive: True to enable recursive lookups, otherwise only use base CMU dictionary
        :param word:
        :return:
        """
        # Word must be fully alphabetic
        if not word.isalpha() or len(word) < 3:
            return None
        self.stat_hits['compound_l2'] += 1  # Register hit

        # Define lookup mode
        def _lu(search_word: str) -> str | None:
            if recursive:
                return self._lookup(search_word, ph_format='sds')
            else:
                return self._cmu_get(search_word)

        # Check if the last part is a single character
        # And that it is repeated in the last char of the first part
        # This is likely silent so remove it
        # i.e. 'Derakk' -> 'Derak'
        # If the word contains a repeated consonant at the end, remove it
        # First check repeated last 2 letters
        if word[-2:][0] == word[-2:][1]:
            # Remove the last char from the word
            word = word[:-1]

        # Holds all matches as tuples
        # (len1, len2, p1, p2, ph1, ph2)
        matches = []

        # Splits the word into every possible combination
        for i in range(1, len(word)):
            p1 = word[:i]
            p2 = word[i:]
            # Looks up both words
            ph1 = _lu(p1)
            if ph1 is None:
                continue  # Skip if not found
            ph2 = _lu(p2)
            if ph2 is None:
                continue  # Skip if not found
            # If both words exist, add to list as tuple
            matches.append((len(p1), len(p2), p1, p2, ph1, ph2))

        # Pick the match with the highest length of both words
        if len(matches) == 0:
            return None
        else:
            # Sort by the minimum of len1 and len2
            matches.sort(key=lambda x: min(x[0], x[1]))
            # Get the highest minimum length match
            match = matches[-1]
            # Otherwise, return the full joined match
            self.stat_resolves['compound_l2'] += 1  # Register resolve
            return match[4] + ' ' + match[5]
