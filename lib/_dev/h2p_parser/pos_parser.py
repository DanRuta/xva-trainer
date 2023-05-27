# Part of Speech Tag Operations

# Method to get the parent part of speech (VERB) or (NOUN) from a pos tag
# from __future__ import annotations

# def get_parent_pos(pos: str) -> str | None:
def get_parent_pos(pos):
    # Get the parent part of speech from a pos tag
    if pos.startswith('VB'):
        return 'VERB'
    elif pos.startswith('NN'):
        return 'NOUN'
    elif pos.startswith('RB'):
        return 'ADVERB'
    else:
        return None

