"""Rebuild annotations.name_uuid from pair_decisions via union-find.

Source of truth is pair_decisions (append-only). annotations.name_uuid is a
materialized view: for each connected component of confirmed-match pairs, we
mint a fresh uuid4 and set it on every annotation in the component. Annotations
with no confirmed matches get name_uuid = NULL.
"""
from __future__ import annotations

from whaleshark_reid.core.schema import RebuildResult, new_name_uuid


class _UnionFind:
    def __init__(self, items: list[str]):
        self.parent: dict[str, str] = {x: x for x in items}
        self.rank: dict[str, int] = {x: 0 for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if y not in self.parent:
            self.parent[y] = y
            self.rank[y] = 0
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def components(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for x in self.parent:
            root = self.find(x)
            out.setdefault(root, []).append(x)
        return out


def rebuild_individuals_cache(storage) -> RebuildResult:
    # Fetch all annotations (need full list so we can null out stale name_uuids)
    all_uuids = storage.list_annotation_uuids()

    # Active match decisions
    match_pairs = storage.list_active_match_pairs()

    uf = _UnionFind(all_uuids)
    for a, b in match_pairs:
        uf.union(a, b)

    components = uf.components()
    # Only components with >= 2 members are "individuals". Singletons get NULL.
    new_name_uuid_by_ann: dict[str, str | None] = {}
    n_components = 0
    n_singletons = 0
    for root, members in components.items():
        if len(members) >= 2:
            n_components += 1
            uuid = new_name_uuid()
            for m in members:
                new_name_uuid_by_ann[m] = uuid
        else:
            n_singletons += 1
            new_name_uuid_by_ann[members[0]] = None

    n_updated = 0
    with storage.transaction():
        for ann_uuid in all_uuids:
            desired = new_name_uuid_by_ann.get(ann_uuid)
            storage.set_annotation_name_uuid(ann_uuid, desired)
            n_updated += 1

    return RebuildResult(
        n_components=n_components,
        n_singletons=n_singletons,
        n_annotations_updated=n_updated,
    )
