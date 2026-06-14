from __future__ import annotations

from rag_practise.transforms.base import QueryTransform, TransformResult
from rag_practise.transforms.baseline import BaselineTransform
from rag_practise.transforms.child_query import ChildQueryTransform
from rag_practise.transforms.expand import ExpandTransform
from rag_practise.transforms.hyde import HyDETransform
from rag_practise.transforms.multi_query import MultiQueryTransform
from rag_practise.transforms.rewrite import RewriteTransform
from rag_practise.transforms.step_back import StepBackTransform

TRANSFORM_REGISTRY = {
    "baseline": BaselineTransform,
    "rewrite": RewriteTransform,
    "expand": ExpandTransform,
    "multi_query": MultiQueryTransform,
    "child_query": ChildQueryTransform,
    "hyde": HyDETransform,
    "step_back": StepBackTransform,
}


def build_transform(name: str) -> QueryTransform:
    try:
        return TRANSFORM_REGISTRY[name]()
    except KeyError as exc:
        known = ", ".join(sorted(TRANSFORM_REGISTRY))
        raise ValueError(f"Unknown transform {name!r}. Known transforms: {known}") from exc


__all__ = [
    "BaselineTransform",
    "ChildQueryTransform",
    "ExpandTransform",
    "HyDETransform",
    "MultiQueryTransform",
    "QueryTransform",
    "RewriteTransform",
    "StepBackTransform",
    "TransformResult",
    "build_transform",
]
