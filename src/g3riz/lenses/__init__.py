"""Lenses — readings of a Film that a study must ask for by name.

A core track is mechanical. A lens is a choice: it needs a threshold, a width,
a confirmation delay or a named rule, and a different reasonable choice gives a
different answer. So a lens is never a default. Nothing in this package is
applied unless a study names it, and the name carries its version.

HOW THIS PACKAGE IS ALLOWED TO GROW
-----------------------------------
A tool moves out of `work/NNN` and into a lens only when a SECOND, independent
study asks for it. Until then it lives with the study that needed it and dies
with it. The library grows from demonstrated repeat demand, never from an
abstraction invented in advance — which is the one habit that made the previous
repository unmaintainable.

A changed rule is a new name, never an edit: `close_inside_v1` and
`close_inside_v2` can both exist, and a result stays readable years later
because it names the one it used. There is no registry object, no code hash and
no provenance framework; the version is the name and the history is git.

WHAT A LENS OWES
----------------
If a study says a lens is decidable at minute `p`, that claim is tested once by
rebuilding on `film.truncate(p)` and requiring the same answer. It is paid at
qualification of the lens, not on every read.
"""

from __future__ import annotations
