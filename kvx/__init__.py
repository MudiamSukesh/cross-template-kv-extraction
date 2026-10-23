"""kvx - cross-template key-value extraction (IE 643 course project).

Formulation: every form describes itself. The model tags each word as part of a
KEY, VALUE, HEADER or nothing, and then links every VALUE to the KEY it answers
(or to "no key"). No field names, queries or schemas are needed, so a template
never seen in training is handled by exactly the same code path.
"""
__version__ = "1.0.0"
