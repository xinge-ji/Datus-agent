from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FieldNode:
    source_system: str
    table: str
    column: str
    table_type: Optional[str] = None
    table_id: Optional[int] = None

    @property
    def key(self) -> str:
        return f"{self.source_system.lower()}.{self.table.lower()}.{self.column.lower()}"

    @staticmethod
    def from_key(key: str) -> "FieldNode":
        """
        解析 source_system.table.column 格式的节点键，table 可能包含 schema，取首尾分段。
        """
        first_dot = key.find(".")
        last_dot = key.rfind(".")
        if first_dot == -1 or last_dot == -1 or first_dot == last_dot:
            raise ValueError(f"非法节点键: {key}")
        source_system = key[:first_dot]
        column = key[last_dot + 1 :]
        table = key[first_dot + 1 : last_dot]
        return FieldNode(source_system=source_system, table=table, column=column)


class UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def add(self, x: str):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.add(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self) -> Dict[str, Set[str]]:
        res: Dict[str, Set[str]] = defaultdict(set)
        for node in list(self.parent.keys()):
            res[self.find(node)].add(node)
        return res


class FieldResolver:
    """
    负责从视图列/等值条件递归追溯到 TABLE/EXTERNAL 的源字段。
    """

    def __init__(
        self,
        feature_map: Dict[str, Dict[str, Any]],
        table_source_index: Dict[Tuple[str, str], Dict[str, Any]],
        default_source_system: str,
    ):
        self.feature_map = {k.lower(): v for k, v in feature_map.items()}
        self.table_source_index = {(k[0].lower(), k[1].lower()): v for k, v in table_source_index.items()}
        self.default_source_system = default_source_system.lower()
        self._table_by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for (ss, tbl), info in self.table_source_index.items():
            self._table_by_name[tbl].append(info | {"source_system": ss})
        self._alias_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def resolve_alias_column(
        self,
        view_name: str,
        feature: Dict[str, Any],
        alias: Optional[str],
        column: Optional[str],
        visited: Optional[Set[Tuple[str, str, str]]] = None,
    ) -> List[FieldNode]:
        """
        从某个视图的 alias.column 追溯到 TABLE/EXTERNAL 字段。
        """
        if not alias or not column:
            return []
        alias_lc = alias.lower()
        col_lc = column.lower()
        view_lc = (view_name or "").lower()
        visited = visited or set()
        key = (view_lc, alias_lc, col_lc)
        if key in visited:
            return []
        visited.add(key)

        alias_map = self._alias_map_for_view(view_lc, feature)
        meta = alias_map.get(alias_lc)
        if not meta:
            return []

        source_type = (meta.get("source_type") or "").upper()
        resolved_name = (meta.get("resolved_name") or meta.get("name") or "").lower()
        source_system = (meta.get("source_system") or self.default_source_system).lower()
        table_type = (meta.get("table_type") or "").upper()
        table_id = meta.get("table_id")

        if source_type in {"TABLE", "EXTERNAL"}:
            return [
                FieldNode(
                    source_system=source_system,
                    table=resolved_name or alias_lc,
                    column=col_lc,
                    table_type=table_type or source_type,
                    table_id=table_id,
                )
            ]

        if source_type != "VIEW":
            return []

        target_feature = self.feature_map.get(resolved_name)
        if not target_feature:
            return []

        return self._resolve_column_in_view(
            view_name=resolved_name,
            feature=target_feature,
            output_column=col_lc,
            visited=visited,
        )

    def resolve_alias(self, view_name: str, feature: Dict[str, Any], alias: Optional[str], column: Optional[str]) -> List[FieldNode]:
        return self.resolve_alias_column(view_name, feature, alias, column, set())

    def resolve_column_feature(self, view_name: str, feature: Dict[str, Any], column_info: Dict[str, Any]) -> List[FieldNode]:
        return self._resolve_column(view_name, feature, column_info, set())

    def _resolve_column_in_view(
        self,
        view_name: str,
        feature: Dict[str, Any],
        output_column: str,
        visited: Set[Tuple[str, str, str]],
    ) -> List[FieldNode]:
        nodes: List[FieldNode] = []
        columns = feature.get("columns") or []
        for col in columns:
            out_name = str(col.get("output_name") or "").lower()
            if out_name != output_column:
                continue
            nodes.extend(self._resolve_column(view_name, feature, col, visited))
        return nodes

    def _resolve_column(
        self,
        view_name: str,
        feature: Dict[str, Any],
        column_info: Dict[str, Any],
        visited: Set[Tuple[str, str, str]],
    ) -> List[FieldNode]:
        nodes: List[FieldNode] = []
        src_alias = column_info.get("source_table_alias")
        src_col = column_info.get("source_column") or column_info.get("output_name")
        if src_alias and src_col:
            nodes.extend(self.resolve_alias_column(view_name, feature, src_alias, src_col, visited))
            return nodes

        ref_cols = column_info.get("referenced_columns") or []
        for ref in ref_cols:
            alias = (ref.get("table_alias") or "").lower() if ref.get("table_alias") else None
            col = (ref.get("column") or "").lower() if ref.get("column") else None
            if alias and col:
                nodes.extend(self.resolve_alias_column(view_name, feature, alias, col, visited))
        return nodes

    def _alias_map_for_view(self, view_name: str, feature: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        if view_name in self._alias_cache:
            return self._alias_cache[view_name]
        mapping: Dict[str, Dict[str, Any]] = {}
        for t in feature.get("tables") or []:
            alias = (t.get("alias") or t.get("resolved_name") or t.get("name") or "").lower()
            resolved = (t.get("resolved_name") or t.get("name") or "").lower()
            source_type = (t.get("source_type") or "").upper()
            source_system = (t.get("source_system") or self.default_source_system).lower()
            table_type = (t.get("table_type") or "").upper()
            table_id = t.get("source_table_id")

            info = self._lookup_table(resolved, source_system)
            if info:
                table_type = table_type or (info.get("table_type") or "").upper()
                table_id = table_id or info.get("table_id")
                source_type = source_type or ("VIEW" if (info.get("table_type") or "").upper() == "VIEW" else "TABLE")
                source_system = (info.get("source_system") or source_system).lower()
            if not source_type:
                source_type = "TABLE"

            mapping[alias] = {
                "alias": alias,
                "resolved_name": resolved,
                "name": t.get("name"),
                "source_type": source_type,
                "source_system": source_system,
                "table_type": table_type or source_type,
                "table_id": table_id,
            }
        self._alias_cache[view_name] = mapping
        return mapping

    def _lookup_table(self, table_name: str, source_system_hint: Optional[str]) -> Optional[Dict[str, Any]]:
        name_lc = (table_name or "").lower()
        if source_system_hint:
            key = (source_system_hint.lower(), name_lc)
            if key in self.table_source_index:
                return self.table_source_index[key]
        candidates = self._table_by_name.get(name_lc) or []
        return candidates[0] if candidates else None
