"""Iterative, explicit-stack fallback for deeply nested, non-cyclic data.

The regular cattrs hooks traverse data through Python function calls, so data
nested deeper than ``sys.getrecursionlimit()`` raises a ``RecursionError``.
The recursion limit is process-wide and raising it from a shared process is
not safe, so instead of recursing this module drives the same built-in
traversal semantics with an explicit, heap-allocated work stack.

Only hooks produced by cattrs itself for composite types (attrs/dataclass
classes, TypedDicts, named tuples, sequences, mappings, sets and tuples) are
driven iteratively; every other hook (user-registered hooks, enums, unions,
literals, ...) is invoked unchanged, so custom behavior is preserved.

The engine keeps no mutable state on the converter: all traversal state lives
in local frame objects and a thread-local activity flag, so a failed
conversion leaves the converter fully usable.
"""

from __future__ import annotations

import threading
from collections import Counter, defaultdict, deque
from collections.abc import Mapping as AbcMapping
from functools import partial
from typing import Any, Tuple, TypeVar

from attrs import NOTHING, Attribute, Converter as AttrsConverter, Factory

from ._compat import (
    ANIES,
    AbcSet,
    Mapping,
    MutableMapping,
    adapted_fields,
    get_args,
    get_full_type_hints,
    get_notrequired_base,
    get_origin,
    has_with_generic,
    is_annotated,
    is_bare,
    is_counter,
    is_deque,
    is_frozenset,
    is_generic,
    is_mapping,
    is_mutable_sequence,
    is_mutable_set,
    is_tuple,
    is_typeddict,
)
from ._generics import deep_copy_with
from .cols import is_abstract_set, is_defaultdict, is_namedtuple
from .errors import (
    AttributeValidationNote,
    ClassValidationError,
    ForbiddenExtraKeysError,
    IterableValidationError,
    IterableValidationNote,
)
from .fns import identity
from .gen._consts import ITERATIVE_MARKER, neutral
from .gen._generics import generate_mapping
from .gen._shared import _annotated_override_or_default, find_structure_handler
from .gen.typeddicts import _adapted_fields as typeddict_adapted_fields
from .gen.typeddicts import _required_keys

__all__ = ["deep_structure", "deep_unstructure", "is_recursion_failure"]

T = TypeVar("T")

#: Thread-local guard ensuring only the outermost conversion retries, and that
#: hooks re-entering the converter while the engine runs stay on the engine.
_state = threading.local()


def is_recursion_failure(exc: BaseException) -> bool:
    """Iteratively check whether *exc* is or contains a ``RecursionError``.

    ``ExceptionGroup`` trees produced by detailed validation can be as deep as
    the data, so this walk must itself never recurse.
    """
    pending = [exc]
    while pending:
        current = pending.pop()
        if isinstance(current, RecursionError):
            return True
        subexcs = getattr(current, "exceptions", None)
        if subexcs:
            pending.extend(subexcs)
    return False


class _Plan:
    """A resolved field/position to process."""

    __slots__ = ("an", "kn", "ian", "t", "handler", "post", "attr")

    def __init__(self, an, kn, ian, t, handler, post, attr):
        #: attribute name
        self.an = an
        #: key to look up in the input mapping
        self.kn = kn
        #: constructor parameter name (attrs alias)
        self.ian = ian
        #: resolved type
        self.t = t
        #: structure/unstructure hook, or ``None`` for passthrough
        self.handler = handler
        #: attrs ``init=False`` field applied after instantiation
        self.post = post
        #: the original attribute
        self.attr = attr


class _Frame:
    """A single node on the explicit work stack."""

    __slots__ = ("parent", "slot")

    def __init__(self, parent, slot):
        self.parent = parent
        #: Describes where the result belongs in the parent frame.
        self.slot = slot

    def next_child(self, engine):
        """Return ``(value, type, slot)`` for the next child, or ``None``."""
        raise NotImplementedError

    def finish(self, engine):
        """Build this frame's result once all children are done."""
        raise NotImplementedError

    def deliver(self, slot, value, engine):
        """Receive a successful child result."""
        raise NotImplementedError

    def fail(self, slot, exc, engine):
        """Receive a failed child. Non-detailed frames re-raise."""
        raise exc


def _resolve_field_type(a: Attribute, mapping: dict[str, Any], cl: type) -> Any:
    """Resolve an attribute type against a generic typevar mapping."""
    t = a.type
    if isinstance(t, TypeVar):
        t = mapping.get(t.__name__, t)
    elif is_generic(t) and not is_bare(t) and not is_annotated(t):
        t = deep_copy_with(t, mapping, cl)
    return t


def _class_generic_mapping(cl: Any) -> tuple[Any, dict[str, Any]]:
    """Mirror the generic setup of the generated class hooks."""
    mapping: dict[str, Any] = {}
    if is_generic(cl):
        base = get_origin(cl)
        mapping = generate_mapping(cl, mapping)
        if base is not None:
            cl = base
    for base in getattr(cl, "__orig_bases__", ()):
        if is_generic(base) and not str(base).startswith("typing.Generic"):
            mapping = generate_mapping(base, mapping)
            break
    return cl, mapping


# ---------------------------------------------------------------------------
# Structure frames
# ---------------------------------------------------------------------------


class _StructClassFrame(_Frame):
    """attrs/dataclass structured from a mapping (AS_DICT strategy)."""

    __slots__ = (
        "cl",
        "plans",
        "index",
        "o",
        "res",
        "post_res",
        "errors",
        "allowed",
        "forbid",
    )

    def __init__(self, parent, slot, o, cl, converter):
        super().__init__(parent, slot)
        self.o = o
        self.res = {}
        self.post_res = {}
        self.errors = []
        self.allowed: set[str] = set()
        self.index = 0

        cl, mapping = _class_generic_mapping(cl)
        self.cl = cl
        attribs = adapted_fields(cl)
        type_overrides = getattr(converter, "type_overrides", {})
        prefer = converter._prefer_attrib_converters
        use_alias = getattr(converter, "use_alias", False)
        self.forbid = getattr(converter, "forbid_extra_keys", False)

        plans = []
        for a in attribs:
            an = a.name
            override = type_overrides.get(a.type, neutral)
            override = _annotated_override_or_default(a.type, override)
            if override.omit:
                continue
            if override.omit is None and not a.init:
                # init=False fields are only handled when they have defaults.
                if a.default is NOTHING:
                    continue
            t = _resolve_field_type(a, mapping, cl)
            if override.struct_hook is not None:
                handler = override.struct_hook
            else:
                handler = find_structure_handler(a, t, converter, prefer)
            if override.rename is not None:
                kn = override.rename
            elif use_alias:
                kn = a.alias
            else:
                kn = an
            self.allowed.add(kn)
            post = not a.init and a.default is not NOTHING
            plans.append(_Plan(an, kn, a.alias, t, handler, post, a))
        self.plans = plans

    def _store(self, plan, value):
        if plan.post:
            self.post_res[plan.an] = value
        else:
            self.res[plan.ian] = value

    def next_child(self, engine):
        while self.index < len(self.plans):
            plan = self.plans[self.index]
            self.index += 1
            if plan.kn not in self.o:
                continue
            raw = self.o[plan.kn]
            if plan.handler is None:
                # No handler: pass the raw value through.
                self._store(plan, raw)
                continue
            return raw, plan.t, plan
        return None

    def deliver(self, slot, value, engine):
        self._store(slot, value)

    def fail(self, slot, exc, engine):
        note = AttributeValidationNote(
            f"Structuring class {self.cl.__qualname__} @ attribute {slot.an}",
            slot.an,
            slot.t,
        )
        exc.__notes__ = [*getattr(exc, "__notes__", []), note]
        self.errors.append(exc)

    def finish(self, engine):
        cl = self.cl
        if self.forbid:
            unknown_fields = set(self.o.keys()) - self.allowed
            if unknown_fields:
                feke = ForbiddenExtraKeysError("", cl, unknown_fields)
                if engine.detailed:
                    self.errors.append(feke)
                else:
                    raise feke
        if engine.detailed and self.errors:
            raise ClassValidationError(
                f"While structuring {cl.__name__!r}", self.errors, cl
            )
        try:
            instance = cl(**self.res)
        except Exception as exc:
            if engine.detailed:
                raise ClassValidationError(
                    f"While structuring {cl.__name__!r}", [exc], cl
                ) from exc
            raise
        if self.post_res:
            for an, value in self.post_res.items():
                if engine.detailed:
                    # Mirror the generated hooks: post-instantiation failures
                    # are collected alongside the other attribute errors.
                    try:
                        setattr(instance, an, value)
                    except Exception as exc:
                        plan = next(p for p in self.plans if p.an == an)
                        note = AttributeValidationNote(
                            f"Structuring class {cl.__qualname__} @ attribute {an}",
                            an,
                            plan.t,
                        )
                        exc.__notes__ = [*getattr(exc, "__notes__", []), note]
                        self.errors.append(exc)
                else:
                    setattr(instance, an, value)
            if engine.detailed and self.errors:
                raise ClassValidationError(
                    f"While structuring {cl.__name__!r}", self.errors, cl
                )
        return instance


class _StructTupleClassFrame(_Frame):
    """attrs/dataclass structured from a tuple (AS_TUPLE strategy)."""

    __slots__ = ("cl", "items", "index", "res", "attribs", "converter")

    def __init__(self, parent, slot, obj, cl, converter):
        super().__init__(parent, slot)
        self.cl = cl
        self.items = obj
        self.res = []
        self.index = 0
        self.converter = converter
        self.attribs = adapted_fields(cl)

    def next_child(self, engine):
        if self.index >= len(self.attribs):
            return None
        a = self.attribs[self.index]
        value = self.items[self.index]
        self.index += 1
        if a.type is None:
            self.res.append(value)
            return None
        return value, a.type, self.index - 1

    def deliver(self, slot, value, engine):
        self.res.append(value)

    def fail(self, slot, exc, engine):
        raise exc

    def finish(self, engine):
        return self.cl(*self.res)


class _StructSeqFrame(_Frame):
    """Homogeneous list/deque/sequence frame."""

    __slots__ = ("cl", "elem_type", "items", "index", "res", "errors", "as_deque")

    def __init__(self, parent, slot, obj, cl, elem_type, as_deque=False):
        super().__init__(parent, slot)
        self.cl = cl
        self.elem_type = elem_type
        self.items = list(obj)
        self.index = 0
        self.res = []
        self.errors = []
        self.as_deque = as_deque

    def next_child(self, engine):
        if self.index >= len(self.items):
            return None
        ix = self.index
        self.index += 1
        return self.items[ix], self.elem_type, ix

    def deliver(self, slot, value, engine):
        self.res.append(value)

    def fail(self, slot, exc, engine):
        note = IterableValidationNote(
            f"Structuring {self.cl} @ index {slot}", slot, self.elem_type
        )
        exc.__notes__ = [*getattr(exc, "__notes__", []), note]
        self.errors.append(exc)

    def finish(self, engine):
        if engine.detailed and self.errors:
            raise IterableValidationError(
                f"While structuring {self.cl!r}", self.errors, self.cl
            )
        if self.as_deque:
            return deque(self.res)
        return self.res


class _StructSetFrame(_Frame):
    """set/frozenset/abstract set frame."""

    __slots__ = (
        "cl",
        "elem_type",
        "items",
        "index",
        "res",
        "errors",
        "structure_to",
    )

    def __init__(self, parent, slot, obj, cl, elem_type, structure_to):
        super().__init__(parent, slot)
        self.cl = cl
        self.elem_type = elem_type
        self.items = list(obj)
        self.index = 0
        self.res = set()
        self.errors = []
        self.structure_to = structure_to

    def next_child(self, engine):
        if self.index >= len(self.items):
            return None
        ix = self.index
        self.index += 1
        return self.items[ix], self.elem_type, ix

    def deliver(self, slot, value, engine):
        self.res.add(value)

    def fail(self, slot, exc, engine):
        e = self.items[slot]
        note = IterableValidationNote(
            f"Structuring {self.structure_to.__name__} @ element {e!r}",
            slot,
            self.elem_type,
        )
        exc.__notes__ = [*getattr(exc, "__notes__", []), note]
        self.errors.append(exc)

    def finish(self, engine):
        if engine.detailed and self.errors:
            raise IterableValidationError(
                f"While structuring {self.cl!r}", self.errors, self.cl
            )
        if self.structure_to is set:
            return self.res
        return self.structure_to(self.res)


class _StructTupleFrame(_Frame):
    """Homogeneous (``tuple[T, ...]``) and heterogeneous tuple frame."""

    __slots__ = ("tup", "types", "items", "index", "res", "errors", "homo")

    def __init__(self, parent, slot, obj, tup):
        super().__init__(parent, slot)
        self.tup = tup
        self.items = list(obj)
        self.index = 0
        self.res = []
        self.errors = []
        args = get_args(tup)
        if args and args[-1] is Ellipsis:
            self.homo = True
            self.types = [args[0]] * len(self.items)
        else:
            self.homo = False
            self.types = list(args)

    def next_child(self, engine):
        if self.index >= len(self.items) or self.index >= len(self.types):
            return None
        ix = self.index
        self.index += 1
        return self.items[ix], self.types[ix], ix

    def deliver(self, slot, value, engine):
        self.res.append(value)

    def fail(self, slot, exc, engine):
        t = self.types[slot]
        note = IterableValidationNote(
            f"Structuring {self.tup} @ index {slot}", slot, t
        )
        exc.__notes__ = [*getattr(exc, "__notes__", []), note]
        self.errors.append(exc)

    def finish(self, engine):
        if not self.homo and len(self.items) != len(self.types):
            problem = "Not enough" if len(self.items) < len(self.types) else "Too many"
            exc = ValueError(
                f"{problem} values in {self.items!r} to structure as {self.tup!r}"
            )
            if engine.detailed:
                exc.__notes__ = [
                    *getattr(exc, "__notes__", []),
                    f"Structuring {self.tup}",
                ]
                self.errors.append(exc)
                raise IterableValidationError(
                    f"While structuring {self.tup!r}", self.errors, self.tup
                )
            raise exc
        if engine.detailed and self.errors:
            raise IterableValidationError(
                f"While structuring {self.tup!r}", self.errors, self.tup
            )
        return tuple(self.res)


class _StructMappingFrame(_Frame):
    """dict/defaultdict/Counter/abstract mapping frame."""

    __slots__ = (
        "cl",
        "key_type",
        "val_type",
        "work",
        "index",
        "res",
        "values",
        "errors",
        "structure_to",
    )

    def __init__(self, parent, slot, obj, cl, converter):
        super().__init__(parent, slot)
        self.cl = cl
        self.res = {}
        self.values = {}
        self.errors = []
        self.index = 0

        args = get_args(cl)
        if len(args) == 2:
            self.key_type, self.val_type = args
        else:
            # Counter-like single-argument mappings.
            self.key_type, self.val_type = args[0], Any

        if is_defaultdict(cl):
            self.structure_to = partial(defaultdict, args[1])
        elif is_counter(cl):
            self.structure_to = Counter
        else:
            origin = get_origin(cl) or cl
            if origin in (AbcMapping, MutableMapping, Mapping):
                self.structure_to = dict
            else:
                self.structure_to = origin

        # Each item produces two children: the value then the key.
        self.work = []
        for i, (k, v) in enumerate(obj.items()):
            self.work.append(("v", i, k, v, self.val_type))
            self.work.append(("k", i, k, k, self.key_type))

    def next_child(self, engine):
        if self.index >= len(self.work):
            return None
        kind, i, raw_key, value, t = self.work[self.index]
        self.index += 1
        return value, t, (kind, i, raw_key)

    def deliver(self, slot, value, engine):
        kind, i, raw_key = slot
        if kind == "v":
            self.values[i] = value
        else:
            self.res[value] = self.values.pop(i)

    def fail(self, slot, exc, engine):
        kind, i, raw_key = slot
        if kind == "v":
            note = IterableValidationNote(
                f"Structuring mapping value @ key {raw_key!r}",
                raw_key,
                self.val_type,
            )
        else:
            note = IterableValidationNote(
                f"Structuring mapping key @ key {raw_key!r}",
                raw_key,
                self.key_type,
            )
        exc.__notes__ = [*getattr(exc, "__notes__", []), note]
        self.errors.append(exc)

    def finish(self, engine):
        if engine.detailed and self.errors:
            raise IterableValidationError(
                f"While structuring {self.cl!r}", self.errors, self.cl
            )
        return self.structure_to(self.res)


class _StructNamedTupleFrame(_Frame):
    """Named tuple structured from an iterable."""

    __slots__ = ("cl", "types", "items", "index", "res", "errors")

    def __init__(self, parent, slot, obj, cl):
        super().__init__(parent, slot)
        self.cl = cl
        self.types = list(get_full_type_hints(cl).values())
        self.items = list(obj)
        self.index = 0
        self.res = []
        self.errors = []

    def next_child(self, engine):
        if self.index >= len(self.items) or self.index >= len(self.types):
            return None
        ix = self.index
        self.index += 1
        return self.items[ix], self.types[ix], ix

    def deliver(self, slot, value, engine):
        self.res.append(value)

    def fail(self, slot, exc, engine):
        note = IterableValidationNote(
            f"Structuring {self.cl} @ index {slot}", slot, self.types[slot]
        )
        exc.__notes__ = [*getattr(exc, "__notes__", []), note]
        self.errors.append(exc)

    def finish(self, engine):
        if len(self.items) != len(self.types):
            problem = "Not enough" if len(self.items) < len(self.types) else "Too many"
            exc = ValueError(
                f"{problem} values in {self.items!r} to structure as {self.cl!r}"
            )
            if engine.detailed:
                exc.__notes__ = [
                    *getattr(exc, "__notes__", []),
                    f"Structuring {self.cl}",
                ]
                self.errors.append(exc)
                raise IterableValidationError(
                    f"While structuring {self.cl!r}", self.errors, self.cl
                )
            raise exc
        if engine.detailed and self.errors:
            raise IterableValidationError(
                f"While structuring {self.cl!r}", self.errors, self.cl
            )
        return self.cl(*self.res)


class _StructTypedDictFrame(_Frame):
    """TypedDict structured from a mapping."""

    __slots__ = (
        "cl",
        "plans",
        "index",
        "o",
        "res",
        "errors",
        "allowed",
        "forbid",
    )

    def __init__(self, parent, slot, o, cl, converter):
        super().__init__(parent, slot)
        self.o = o
        self.res = o.copy()
        self.errors = []
        self.allowed: set[str] = set()
        self.index = 0

        origin = get_origin(cl)
        if is_generic(cl) and origin is not None:
            cl = origin
        self.cl = cl
        attribs = typeddict_adapted_fields(cl)
        required = _required_keys(cl)
        self.forbid = getattr(converter, "forbid_extra_keys", False)

        if converter.detailed_validation and not isinstance(o, AbcMapping):
            raise ClassValidationError(
                f"While structuring {cl.__name__!r}",
                [TypeError(f"expected a mapping, not {o.__class__.__name__}")],
                cl,
            )

        plans = []
        for a in attribs:
            an = a.name
            override = _annotated_override_or_default(a.type, neutral)
            if override.omit:
                continue
            t = a.type
            nrb = get_notrequired_base(t)
            if nrb is not NOTHING:
                t = nrb
            if override.struct_hook is not None:
                handler = override.struct_hook
            else:
                handler = find_structure_handler(a, t, converter)
            kn = an if override.rename is None else override.rename
            self.allowed.add(kn)
            plans.append((an, kn, t, handler, an in required))
        self.plans = plans

    def next_child(self, engine):
        while self.index < len(self.plans):
            an, kn, t, handler, required = self.plans[self.index]
            self.index += 1
            if not required and kn not in self.o:
                continue
            if handler is None:
                if kn != an:
                    self.res[an] = self.o[kn]
                continue
            return self.o[kn], t, (an, kn, t)
        return None

    def deliver(self, slot, value, engine):
        an, kn, t = slot
        self.res[an] = value
        if kn != an:
            self.res.pop(kn, None)

    def fail(self, slot, exc, engine):
        an, kn, t = slot
        note = AttributeValidationNote(
            f"Structuring typeddict {self.cl.__qualname__} @ attribute {an}",
            an,
            t,
        )
        exc.__notes__ = [*getattr(exc, "__notes__", []), note]
        self.errors.append(exc)

    def finish(self, engine):
        cl = self.cl
        if self.forbid:
            unknown_fields = self.o.keys() - self.allowed
            if unknown_fields:
                feke = ForbiddenExtraKeysError("", cl, set(unknown_fields))
                if engine.detailed:
                    self.errors.append(feke)
                else:
                    raise feke
        if engine.detailed and self.errors:
            raise ClassValidationError(
                f"While structuring {cl.__name__!r}", self.errors, cl
            )
        return self.res


# ---------------------------------------------------------------------------
# Unstructure frames
# ---------------------------------------------------------------------------


class _UnstructClassFrame(_Frame):
    """attrs/dataclass unstructured to a dict (AS_DICT) or tuple (AS_TUPLE)."""

    __slots__ = (
        "cl",
        "plans",
        "index",
        "instance",
        "res",
        "as_tuple",
        "dict_factory",
        "omit_if_default",
        "use_alias",
    )

    def __init__(self, parent, slot, instance, cl, converter, as_tuple):
        super().__init__(parent, slot)
        self.instance = instance
        self.as_tuple = as_tuple
        self.index = 0
        self.res = [] if as_tuple else converter._dict_factory()
        self.dict_factory = converter._dict_factory
        self.omit_if_default = getattr(converter, "omit_if_default", False)
        self.use_alias = getattr(converter, "use_alias", False)

        if as_tuple:
            # BaseConverter.unstructure_attrs_astuple semantics.
            self.cl = cl
            self.plans = [
                _Plan(a.name, a.name, a.alias, a.type, None, False, a)
                for a in adapted_fields(cl)
            ]
            return

        cl, mapping = _class_generic_mapping(cl)
        self.cl = cl
        attribs = adapted_fields(cl)
        type_overrides = getattr(converter, "type_overrides", {})
        plans = []
        for a in attribs:
            an = a.name
            override = type_overrides.get(a.type, neutral)
            override = _annotated_override_or_default(a.type, override)
            if override.omit:
                continue
            if override.omit is None and not a.init:
                continue
            t = _resolve_field_type(a, mapping, cl)
            if override.unstruct_hook is not None:
                handler = override.unstruct_hook
            else:
                if a.type is None:
                    handler = converter.unstructure
                else:
                    try:
                        handler = converter.get_unstructure_hook(
                            t, cache_result=False
                        )
                    except RecursionError:
                        handler = converter.unstructure
            if override.rename is not None:
                kn = override.rename
            elif self.use_alias:
                kn = a.alias
            else:
                kn = an
            plans.append(_Plan(an, kn, a.alias, t, handler, False, a))
        self.plans = plans

    def _is_omitted(self, plan, value):
        a = plan.attr
        d = a.default
        if d is NOTHING:
            return False
        override = _annotated_override_or_default(a.type, neutral)
        if not (
            (self.omit_if_default and override.omit_if_default is not False)
            or override.omit_if_default
        ):
            return False
        if isinstance(d, Factory):
            default = d.factory(self.instance) if d.takes_self else d.factory()
        else:
            default = d
        conv = a.converter
        if conv is not None:
            if isinstance(conv, AttrsConverter):
                default = conv(default, self.instance, a)
            elif isinstance(d, Factory):
                default = conv(default)
            else:
                default = conv(d)
        return value == default

    def next_child(self, engine):
        while self.index < len(self.plans):
            plan = self.plans[self.index]
            self.index += 1
            value = getattr(self.instance, plan.an)
            if not self.as_tuple and self._is_omitted(plan, value):
                continue
            if self.as_tuple:
                # AS_TUPLE dispatches on the declared type or runtime class.
                t = plan.t if plan.t is not None else value.__class__
                plan.t = t
                return value, t, plan
            if plan.handler is None or plan.handler == identity:
                self.res[plan.kn] = value
                continue
            return value, plan.t, plan
        return None

    def deliver(self, slot, value, engine):
        if self.as_tuple:
            self.res.append(value)
        else:
            self.res[slot.kn] = value

    def finish(self, engine):
        if self.as_tuple:
            return tuple(self.res)
        return self.res


class _UnstructSeqFrame(_Frame):
    """list/deque/sequence/set/frozenset unstructure frame."""

    __slots__ = ("cl", "elem_type", "items", "index", "res", "target", "runtime")

    def __init__(self, parent, slot, obj, cl, elem_type, target, runtime):
        super().__init__(parent, slot)
        self.cl = cl
        self.elem_type = elem_type
        self.items = list(obj)
        self.index = 0
        self.res = []
        self.target = target
        #: dispatch on the runtime element class when the type is bare/Any
        self.runtime = runtime

    def next_child(self, engine):
        if self.index >= len(self.items):
            return None
        ix = self.index
        self.index += 1
        value = self.items[ix]
        if self.runtime:
            return value, value.__class__, ix
        return value, self.elem_type, ix

    def deliver(self, slot, value, engine):
        self.res.append(value)

    def finish(self, engine):
        return self.target(self.res)


class _UnstructMappingFrame(_Frame):
    """mapping unstructure frame."""

    __slots__ = (
        "cl",
        "key_type",
        "val_type",
        "work",
        "index",
        "res",
        "values",
        "target",
        "runtime",
    )

    def __init__(self, parent, slot, obj, cl, target, runtime_key, runtime_val):
        super().__init__(parent, slot)
        self.cl = cl
        self.res = {}
        self.values = {}
        self.index = 0
        self.target = target
        args = get_args(cl)
        if len(args) == 2:
            self.key_type, self.val_type = args
        else:
            self.key_type, self.val_type = args[0], Any
        self.work = []
        for i, (k, v) in enumerate(obj.items()):
            kt = k.__class__ if runtime_key else self.key_type
            vt = v.__class__ if runtime_val else self.val_type
            self.work.append(("v", i, k, v, vt))
            self.work.append(("k", i, k, k, kt))

    def next_child(self, engine):
        if self.index >= len(self.work):
            return None
        kind, i, raw_key, value, t = self.work[self.index]
        self.index += 1
        return value, t, (kind, i, raw_key)

    def deliver(self, slot, value, engine):
        kind, i, raw_key = slot
        if kind == "v":
            self.values[i] = value
        else:
            self.res[value] = self.values.pop(i)

    def finish(self, engine):
        return self.target(self.res)


class _UnstructTupleFrame(_Frame):
    """tuple/hetero tuple unstructure frame."""

    __slots__ = ("tup", "types", "items", "index", "res", "target", "runtime")

    def __init__(self, parent, slot, obj, tup, target):
        super().__init__(parent, slot)
        self.tup = tup
        self.items = list(obj)
        self.index = 0
        self.res = []
        self.target = target
        args = get_args(tup)
        if not args or tup in (Tuple, tuple):
            self.runtime = True
            self.types = []
        elif args[-1] is Ellipsis:
            self.runtime = False
            self.types = [args[0]] * len(self.items)
        else:
            self.runtime = False
            self.types = list(args)

    def next_child(self, engine):
        if self.index >= len(self.items):
            return None
        ix = self.index
        self.index += 1
        value = self.items[ix]
        if self.runtime:
            return value, value.__class__, ix
        return value, self.types[ix], ix

    def deliver(self, slot, value, engine):
        self.res.append(value)

    def finish(self, engine):
        return self.target(self.res)


class _UnstructNamedTupleFrame(_Frame):
    """named tuple unstructure frame."""

    __slots__ = ("cl", "types", "items", "index", "res", "target")

    def __init__(self, parent, slot, obj, cl, target):
        super().__init__(parent, slot)
        self.cl = cl
        self.types = list(get_full_type_hints(cl).values())
        self.items = list(obj)
        self.index = 0
        self.res = []
        self.target = target

    def next_child(self, engine):
        if self.index >= len(self.items):
            return None
        ix = self.index
        self.index += 1
        return self.items[ix], self.types[ix], ix

    def deliver(self, slot, value, engine):
        self.res.append(value)

    def finish(self, engine):
        return self.target(self.res)


class _UnstructTypedDictFrame(_Frame):
    """TypedDict unstructure frame."""

    __slots__ = ("cl", "plans", "index", "instance", "res")

    def __init__(self, parent, slot, instance, cl, converter):
        super().__init__(parent, slot)
        self.instance = instance
        self.res = instance.copy()
        self.index = 0

        origin = get_origin(cl)
        if is_generic(cl) and origin is not None:
            cl = origin
        self.cl = cl
        attribs = typeddict_adapted_fields(cl)
        required = _required_keys(cl)
        plans = []
        for a in attribs:
            an = a.name
            override = _annotated_override_or_default(a.type, neutral)
            if override.omit:
                self.res.pop(an, None)
                continue
            t = a.type
            nrb = get_notrequired_base(t)
            if nrb is not NOTHING:
                t = nrb
            if override.unstruct_hook is not None:
                handler = override.unstruct_hook
            else:
                try:
                    handler = converter.get_unstructure_hook(t)
                except RecursionError:
                    handler = converter.unstructure
            kn = an if override.rename is None else override.rename
            plans.append((an, kn, t, handler, an in required, override.rename))
        self.plans = plans

    def next_child(self, engine):
        while self.index < len(self.plans):
            an, kn, t, handler, required, rename = self.plans[self.index]
            self.index += 1
            if not required and an not in self.instance:
                continue
            if handler == identity:
                if rename is not None:
                    self.res[kn] = self.res.pop(an)
                continue
            if rename is not None:
                self.res.pop(an, None)
            return self.instance[an], t, (an, kn)

    def deliver(self, slot, value, engine):
        an, kn = slot
        self.res[kn] = value

    def finish(self, engine):
        return self.res


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


class _StructEngine:
    __slots__ = ("c", "detailed", "stack")

    def __init__(self, converter):
        self.c = converter
        self.detailed = converter.detailed_validation
        self.stack: list[_Frame] = []

    def leaf(self, value, t):
        return self.c._structure_func.dispatch(t)(value, t)

    def build_frame(self, value, t, parent, slot):
        c = self.c
        hook = c._structure_func.dispatch(t)
        if not getattr(hook, ITERATIVE_MARKER, False):
            return None
        if is_typeddict(t):
            return _StructTypedDictFrame(parent, slot, value, t, c)
        if is_namedtuple(t):
            return _StructNamedTupleFrame(parent, slot, value, t)
        if has_with_generic(t):
            if c.unstruct_strat.value == "astuple":
                return _StructTupleClassFrame(parent, slot, value, get_origin(t) or t, c)
            return _StructClassFrame(parent, slot, value, t, c)
        if is_tuple(t):
            if t in (Tuple, tuple) or (
                get_args(t) and get_args(t)[-1] is Ellipsis and get_args(t)[0] in ANIES
            ):
                return None
            return _StructTupleFrame(parent, slot, value, t)
        if is_deque(t):
            if is_bare(t) or t.__args__[0] in ANIES:
                return None
            return _StructSeqFrame(parent, slot, value, t, t.__args__[0], True)
        if is_mutable_sequence(t):
            if is_bare(t) or t.__args__[0] in ANIES:
                return None
            return _StructSeqFrame(parent, slot, value, t, t.__args__[0])
        if is_mutable_set(t):
            if is_bare(t) or t.__args__[0] in ANIES:
                return None
            return _StructSetFrame(parent, slot, value, t, t.__args__[0], set)
        if is_frozenset(t):
            if is_bare(t) or t.__args__[0] in ANIES:
                return None
            return _StructSetFrame(parent, slot, value, t, t.__args__[0], frozenset)
        if is_abstract_set(t):
            args = get_args(t)
            if not args or args[0] in ANIES:
                return None
            return _StructSetFrame(parent, slot, value, t, args[0], frozenset)
        if is_mapping(t):
            if is_bare(t) or get_args(t) == (Any, Any):
                return None
            return _StructMappingFrame(parent, slot, value, t, c)
        return None

    def run(self, first):
        self.stack.append(first)
        while self.stack:
            frame = self.stack[-1]
            desc = frame.next_child(self)
            if desc is None:
                self.stack.pop()
                try:
                    result = frame.finish(self)
                except Exception as exc:
                    if frame.parent is None:
                        raise
                    frame.parent.fail(frame.slot, exc, self)
                    continue
                if frame.parent is None:
                    return result
                frame.parent.deliver(frame.slot, result, self)
            else:
                value, t, child_slot = desc
                child = self.build_frame(value, t, frame, child_slot)
                if child is None:
                    try:
                        result = self.leaf(value, t)
                    except Exception as exc:
                        if not self.detailed:
                            raise
                        frame.fail(child_slot, exc, self)
                    else:
                        frame.deliver(child_slot, result, self)
                else:
                    self.stack.append(child)
        raise RuntimeError("unreachable")  # pragma: no cover


class _UnstructEngine:
    __slots__ = ("c", "detailed", "stack")

    def __init__(self, converter):
        self.c = converter
        self.detailed = converter.detailed_validation
        self.stack: list[_Frame] = []

    def leaf(self, value, t):
        return self.c._unstructure_func.dispatch(t)(value)

    def build_frame(self, value, t, parent, slot):
        c = self.c
        hook = c._unstructure_func.dispatch(t)
        if not getattr(hook, ITERATIVE_MARKER, False):
            return None
        overrides = getattr(c, "_unstruct_collection_overrides", None)
        as_tuple = c.unstruct_strat.value == "astuple"
        if is_typeddict(t):
            return _UnstructTypedDictFrame(parent, slot, value, t, c)
        if is_namedtuple(t):
            target = tuple
            if overrides is not None:
                target = overrides.get(tuple, tuple)
            return _UnstructNamedTupleFrame(parent, slot, value, t, target)
        if has_with_generic(t):
            origin = get_origin(t) or t
            return _UnstructClassFrame(
                parent, slot, value, origin if as_tuple else t, c, as_tuple
            )
        if is_tuple(t):
            if overrides is not None:
                target = overrides.get(get_origin(t) or t, tuple)
            else:
                target = tuple
            return _UnstructTupleFrame(parent, slot, value, t, target)
        if is_deque(t):
            args = get_args(t)
            runtime = not args or args[0] in ANIES
            if overrides is not None:
                target = overrides.get(deque, list)
            else:
                target = value.__class__
            elem = None if runtime else args[0]
            return _UnstructSeqFrame(parent, slot, value, t, elem, target, runtime)
        if is_mutable_sequence(t):
            args = get_args(t)
            runtime = not args or args[0] in ANIES
            if overrides is not None:
                origin = get_origin(t) or t
                target = overrides.get(origin, list)
            else:
                target = value.__class__
            elem = None if runtime else args[0]
            return _UnstructSeqFrame(parent, slot, value, t, elem, target, runtime)
        if is_mutable_set(t):
            args = get_args(t)
            runtime = not args or args[0] in ANIES
            if overrides is not None:
                target = overrides.get(set, set)
            else:
                target = value.__class__
            return _UnstructSeqFrame(
                parent, slot, value, t, None if runtime else args[0], target, runtime
            )
        if is_frozenset(t):
            args = get_args(t)
            runtime = not args or args[0] in ANIES
            if overrides is not None:
                target = overrides.get(frozenset, frozenset)
            else:
                target = value.__class__
            return _UnstructSeqFrame(
                parent, slot, value, t, None if runtime else args[0], target, runtime
            )
        if is_abstract_set(t):
            args = get_args(t)
            runtime = not args or args[0] in ANIES
            target = overrides.get(AbcSet, set) if overrides is not None else set
            return _UnstructSeqFrame(
                parent, slot, value, t, None if runtime else args[0], target, runtime
            )
        if is_mapping(t):
            args = get_args(t)
            runtime_key = is_bare(t) or not args or args[0] in ANIES
            runtime_val = (
                is_bare(t) or len(args) < 2 or args[1] in ANIES
            )
            if overrides is not None:
                origin = get_origin(t) or t
                if origin in (AbcMapping, MutableMapping, Mapping):
                    origin = dict
                target = overrides.get(origin, dict if is_counter(t) is False else dict)
            else:
                target = value.__class__
            return _UnstructMappingFrame(
                parent, slot, value, t, target, runtime_key, runtime_val
            )
        return None

    def run(self, first):
        self.stack.append(first)
        while self.stack:
            frame = self.stack[-1]
            desc = frame.next_child(self)
            if desc is None:
                self.stack.pop()
                result = frame.finish(self)
                if frame.parent is None:
                    return result
                frame.parent.deliver(frame.slot, result, self)
            else:
                value, t, child_slot = desc
                child = self.build_frame(value, t, frame, child_slot)
                if child is None:
                    frame.deliver(child_slot, self.leaf(value, t), self)
                else:
                    self.stack.append(child)
        raise RuntimeError("unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def deep_structure(converter, obj: Any, cl: Any) -> Any:
    """Structure *obj* into *cl* using an explicit stack."""
    was = getattr(_state, "active", False)
    _state.active = True
    try:
        engine = _StructEngine(converter)
        frame = engine.build_frame(obj, cl, None, None)
        if frame is None:
            return converter._structure_func.dispatch(cl)(obj, cl)
        return engine.run(frame)
    finally:
        _state.active = was


def deep_unstructure(converter, obj: Any, unstructure_as: Any = None) -> Any:
    """Unstructure *obj* using an explicit stack."""
    was = getattr(_state, "active", False)
    _state.active = True
    try:
        t = obj.__class__ if unstructure_as is None else unstructure_as
        engine = _UnstructEngine(converter)
        frame = engine.build_frame(obj, t, None, None)
        if frame is None:
            return converter._unstructure_func.dispatch(t)(obj)
        return engine.run(frame)
    finally:
        _state.active = was
