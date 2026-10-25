"""LinkedIn drafts: one checked draft from a stored item and the owner's own point of view.

The model writes one post per click; code then proves every number, quote and link in it came from
the stored article or from the owner's own words, and every first-person claim was his to make.
The flags are notes on the saved draft, never edits: what the model wrote is what is stored
(ADR 0015). The point of view is stored with its draft and is never logged at any level.
"""
