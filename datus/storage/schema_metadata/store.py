# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import csv
from io import StringIO

import pyarrow as pa
from lancedb.rerankers import Reranker

from datus.configuration.agent_config import AgentConfig
from datus.schemas.base import TABLE_TYPE
from datus.schemas.node_models import ColumnSearchResult, TableSchema, TableValue
from datus.storage.base import BaseEmbeddingStore, WhereExpr
from datus.storage.embedding_models import EmbeddingModel
from datus.storage.lancedb_conditions import Node, and_, build_where, eq, or_
from datus.utils.constants import DBType
from datus.utils.json_utils import json2csv
from datus.utils.loggings import get_logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger(__name__)


class BaseMetadataStorage(BaseEmbeddingStore):
    """
    Base class for metadata storage, include table, view and materialized view(abbreviated as mv).
    properties:
        - db_path: str, database path to store the metadata
        - embedding_model: EmbeddingModel, embedding model to embed the metadata
        - table_name: str, table name to store the metadata
        - vector_source_name: str, vector source name, required, should define in subclass
        - reranker: Reranker, reranker, optional

    schema properties:
        - identifier: str, unique identifier for the metadata, spliced by catalog_name, database_name, schema_name,
        table_name, table_type
        - catalog_name: str, catalog name, optional
        - database_name: str, database name, optional
        - schema_name: str, schema name, optional
        - table_name: str, table name, required
        - table_type: str, table type, choices: table, view, mv
        - vector_source_name: str, vector source name, required
        - vector: list[float], vector, required
    """

    def __init__(
        self,
        db_path: str,
        embedding_model: EmbeddingModel,
        table_name: str,
        vector_source_name: str,
        schema: Optional[pa.Schema] = None,
    ):
        if schema is None:
            schema = pa.schema(
                [
                    pa.field("identifier", pa.string()),
                    pa.field("catalog_name", pa.string()),
                    pa.field("database_name", pa.string()),
                    pa.field("schema_name", pa.string()),
                    pa.field("table_name", pa.string()),
                    pa.field("table_type", pa.string()),
                    pa.field(vector_source_name, pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
                ]
            )
        super().__init__(
            db_path=db_path,
            table_name=table_name,
            embedding_model=embedding_model,
            schema=schema,
            vector_source_name=vector_source_name,
        )
        self.reranker = None

    def search_similar(
        self,
        query_text: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        top_n: int = 5,
        table_type: TABLE_TYPE = "table",
        reranker: Optional[Reranker] = None,
    ) -> pa.Table:
        where = _build_where_clause(
            catalog_name=catalog_name, database_name=database_name, schema_name=schema_name, table_type=table_type
        )
        return self.do_search_similar(query_text, top_n=top_n, where=where, reranker=reranker)

    def do_search_similar(
        self,
        query_text: str,
        top_n: int = 5,
        where: WhereExpr = None,
        reranker: Optional[Reranker] = None,
    ) -> pa.Table:
        return self.search(
            query_text,
            top_n=top_n,
            where=where,
            reranker=reranker,
        )

    def create_indices(self):
        # Ensure table is ready before creating indices
        self._ensure_table_ready()

        # create scalar index
        try:
            self.table.create_scalar_index("database_name", replace=True)
            self.table.create_scalar_index("catalog_name", replace=True)
            self.table.create_scalar_index("schema_name", replace=True)
            self.table.create_scalar_index("table_name", replace=True)
            self.table.create_scalar_index("table_type", replace=True)
        except Exception as e:
            logger.warning(f"Failed to create scalar index for {self.table_name} table: {str(e)}")

        # self.create_vector_index()
        self.create_fts_index(["database_name", "schema_name", "table_name", self.vector_source_name])

    def search_all(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_type: TABLE_TYPE = "full",
        select_fields: Optional[List[str]] = None,
    ) -> pa.Table:
        """Search all schemas for a given database name."""
        # Ensure table is ready before searching
        self._ensure_table_ready()

        where = _build_where_clause(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_type=table_type,
        )
        return self._search_all(where=where, select_fields=select_fields)


class SchemaStorage(BaseMetadataStorage):
    """Store and manage schema lineage data in LanceDB."""

    def __init__(self, db_path: str, embedding_model: EmbeddingModel):
        """Initialize the schema store.

        Args:
            db_path: Path to the LanceDB database directory
        """
        super().__init__(
            db_path=db_path,
            table_name="schema_metadata",
            embedding_model=embedding_model,
            vector_source_name="definition",
        )
        self.reranker = None
        # self.reranker = CrossEncoderReranker(
        #     model_name="BAAI/bge-reranker-large", device=get_device(), column="schema_text"
        # )

    def _extract_table_name(self, schema_text: str) -> str:
        """Extract table name from CREATE TABLE statement."""
        # Simple extraction - can be enhanced for more complex cases
        words = schema_text.split()
        if len(words) >= 3 and words[0].upper() == "CREATE" and words[1].upper() == "TABLE":
            return words[2].strip("()").strip()
        return ""

    def search_all_schemas(self, database_name: str = "", catalog_name: str = "") -> Set[str]:
        search_result = self._search_all(
            where=_build_where_clause(database_name=database_name, catalog_name=catalog_name),
            select_fields=["schema_name"],
        )
        return {search_result["schema_name"]}

    def search_top_tables_by_every_schema(
        self,
        query_text: str,
        database_name: str = "",
        catalog_name: str = "",
        all_schemas: Optional[Set[str]] = None,
        top_n: int = 20,
    ) -> pa.Table:
        if all_schemas is None:
            all_schemas = self.search_all_schemas(catalog_name=catalog_name, database_name=database_name)
        result = []
        for schema in all_schemas:
            result.append(
                self.search_similar(
                    query_text=query_text,
                    database_name=database_name,
                    catalog_name=catalog_name,
                    schema_name=schema,
                    top_n=top_n,
                )
            )
        return pa.concat_tables(result, promote_options="default")

    def get_schema(
        self, table_name: str, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> pa.Table:
        where = _build_where_clause(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_type="full",
        )
        table_condition = eq("table_name", table_name)
        if where:
            where_condition = and_(where, table_condition)
        else:
            where_condition = table_condition
        where_clause = build_where(where_condition)
        return (
            self.table.search()
            .where(where_clause)
            .select(["catalog_name", "database_name", "schema_name", "table_name", "table_type", "definition"])
            .to_arrow()
        )


class SchemaValueStorage(BaseMetadataStorage):
    def __init__(self, db_path: str, embedding_model: EmbeddingModel):
        super().__init__(
            db_path=db_path,
            embedding_model=embedding_model,
            table_name="schema_value",
            vector_source_name="sample_rows",
            schema=pa.schema(
                [
                    pa.field("identifier", pa.string()),
                    pa.field("catalog_name", pa.string()),
                    pa.field("database_name", pa.string()),
                    pa.field("schema_name", pa.string()),
                    pa.field("table_name", pa.string()),
                    pa.field("table_type", pa.string()),
                    pa.field("column_name", pa.string()),
                    pa.field("sample_rows", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
                ]
            ),
        )
        self.reranker = None

    def create_indices(self):
        self._ensure_table_ready()
        try:
            self.table.create_scalar_index("database_name", replace=True)
            self.table.create_scalar_index("catalog_name", replace=True)
            self.table.create_scalar_index("schema_name", replace=True)
            self.table.create_scalar_index("table_name", replace=True)
            self.table.create_scalar_index("table_type", replace=True)
            self.table.create_scalar_index("column_name", replace=True)
        except Exception as e:
            logger.warning(f"Failed to create scalar index for {self.table_name} table: {str(e)}")

        self.create_vector_index()
        self.create_fts_index([
            "database_name",
            "schema_name",
            "table_name",
            "column_name",
            self.vector_source_name,
        ])

    def search_columns_by_keyword(
        self,
        query_text: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_type: TABLE_TYPE = "table",
        top_n: int = 10,
    ) -> pa.Table:
        where = _build_where_clause(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_type=table_type,
        )
        return self.search(
            query_text,
            top_n=top_n,
            where=where,
            select_fields=[
                "identifier",
                "catalog_name",
                "database_name",
                "schema_name",
                "table_name",
                "table_type",
                "column_name",
                "sample_rows",
            ],
        )


class SchemaWithValueRAG:
    def __init__(
        self,
        agent_config: AgentConfig,
        sub_agent_name: Optional[str] = None
        # use_rerank: bool = False,
    ):
        from datus.storage.cache import get_storage_cache_instance

        self.schema_store = get_storage_cache_instance(agent_config).schema_storage(sub_agent_name)
        self.value_store = get_storage_cache_instance(agent_config).schema_value_storage(sub_agent_name)

    def store_batch(self, schemas: List[Dict[str, Any]], values: List[Dict[str, Any]]):
        # Process schemas and values in batches of 500
        # batch_size = 500
        if schemas:
            self.schema_store.store_batch(schemas)

        if len(values) == 0:
            return

        column_type_map = self._build_column_type_map(schemas)

        final_values = []
        for item in values:
            if "sample_rows" not in item or not item["sample_rows"]:
                continue
            sample_rows = item["sample_rows"]
            if isinstance(sample_rows, list):
                sample_rows = json2csv(sample_rows)
            item["sample_rows"] = sample_rows
            final_values.extend(self._expand_sample_rows_by_column(item, column_type_map))
        self.value_store.store_batch(final_values)

        logger.debug(f"Batch stored {len(schemas)} schemas, {len(final_values)} values")

    def after_init(self):
        """After init the schema and value, create the indices for the tables."""
        self.schema_store.create_indices()
        self.value_store.create_indices()

    def get_schema_size(self):
        return self.schema_store.table_size()

    def get_value_size(self):
        return self.value_store.table_size()

    def search_similar(
        self,
        query_text: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        use_rerank: bool = False,
        table_type: TABLE_TYPE = "table",
        top_n: int = 5,
    ) -> Tuple[pa.Table, pa.Table]:
        where = _build_where_clause(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_type=table_type,
        )
        schema_results = self.schema_store.do_search_similar(
            query_text,
            top_n=top_n,
            where=where,
            reranker=self.schema_store.reranker if use_rerank else None,
        )
        value_results = self.value_store.do_search_similar(
            query_text,
            top_n=top_n,
            where=where,
            reranker=self.value_store.reranker if use_rerank else None,
        )
        return schema_results, value_results

    def search_all_schemas(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_type: TABLE_TYPE = "full",
        select_fields: Optional[List[str]] = None,
    ) -> pa.Table:
        """Search all schemas for a given database name.
        Args:
            database_name: The catalog name to search for. If not provided, search all catalogs.
            catalog_name:  The database name to search for. If not provided, search all databases.
            schema_name: The schema name to search for. If not provided, search all schemas.
            table_type: The table type to search for.
            select_fields: The fields to search for. If not provided, search all fields.

        Returns:
            A list of dictionaries containing the schema information.
        """
        return self.schema_store.search_all(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_type=table_type,
            select_fields=select_fields,
        )

    def search_all_value(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_type: TABLE_TYPE = "full"
    ) -> pa.Table:
        """Search all schemas for a given database name.
        :param database_name: The catalog name to search for. If not provided, search all catalogs.
        :param catalog_name:  The database name to search for. If not provided, search all databases.
        :param schema_name: The schema name to search for. If not provided, search all schemas.
        :param table_type: The table type to search for.
        Returns:
            A list of dictionaries containing the schema information.
        """
        return self.value_store.search_all(
            catalog_name=catalog_name, database_name=database_name, schema_name=schema_name, table_type=table_type
        )

    def search_columns(
        self,
        query_text: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_type: TABLE_TYPE = "table",
        top_n: int = 10,
        ) -> pa.Table:
        return self.value_store.search_columns_by_keyword(
            query_text=query_text,
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_type=table_type,
            top_n=top_n,
        )

    def search_columns_by_keywords(
        self,
        keywords: List[str],
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_type: TABLE_TYPE = "table",
        top_n: int = 5,
    ) -> List[ColumnSearchResult]:
        """Search columns for a list of keywords and return structured hints."""

        hints: List[ColumnSearchResult] = []
        seen: Set[tuple[str, str]] = set()

        for keyword in keywords:
            sanitized_keyword = keyword.strip()
            if not sanitized_keyword:
                continue
            column_table = self.search_columns(
                query_text=sanitized_keyword,
                catalog_name=catalog_name,
                database_name=database_name,
                schema_name=schema_name,
                table_type=table_type,
                top_n=top_n,
            )
            for result in ColumnSearchResult.from_arrow(column_table, sanitized_keyword):
                key = (result.table_identifier, result.column_name)
                if key in seen:
                    continue
                seen.add(key)
                hints.append(result)
        return hints

    def search_tables(
        self,
        tables: list[str],
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        dialect: str = DBType.SQLITE,
    ) -> Tuple[List[TableSchema], List[TableValue]]:
        """
        Search schemas and values for given table names.
        """
        # Ensure tables are ready before direct table access
        self.schema_store._ensure_table_ready()
        self.value_store._ensure_table_ready()

        # Parse table names and build where clause
        table_conditions = []
        for full_table in tables:
            parts = full_table.split(".")
            table_name = parts[-1]
            if len(parts) == 4:
                cat, db, sch = parts[0], parts[1], parts[2]
                table_conditions.append(
                    _build_where_clause(
                        table_name=table_name,
                        catalog_name=cat,
                        database_name=db,
                        schema_name=sch,
                        table_type="full",
                    )
                )
            elif len(parts) == 3:
                # Format: database_name.schema_name.table_name
                if dialect == DBType.STARROCKS:
                    cat, db, sch = parts[0], parts[1], ""
                else:
                    cat, db, sch = catalog_name, parts[0], parts[1]

                table_conditions.append(
                    _build_where_clause(
                        table_name=table_name,
                        catalog_name=cat,
                        database_name=db,
                        schema_name=sch,
                        table_type="full",
                    )
                )
            elif len(parts) == 2:
                # Format: database_name.table_name(Maybe need fix for other dialects)
                if dialect in (DBType.SQLITE, DBType.MYSQL, DBType.STARROCKS):
                    cat, db, sch = catalog_name, parts[0], ""
                else:
                    cat, db, sch = catalog_name, database_name, parts[0]

                table_conditions.append(
                    _build_where_clause(
                        table_name=table_name,
                        catalog_name=cat,
                        database_name=db,
                        schema_name=sch,
                        table_type="full",
                    )
                )
            else:
                table_conditions.append(
                    _build_where_clause(
                        table_name=table_name,
                        catalog_name=catalog_name,
                        database_name=database_name,
                        schema_name=schema_name,
                        table_type="full",
                    )
                )

        if table_conditions:
            combined_condition = table_conditions[0] if len(table_conditions) == 1 else or_(*table_conditions)
            where_clause = build_where(combined_condition)
            schema_query = self.schema_store.table.search().where(where_clause)
            value_query = self.value_store.table.search().where(where_clause)
        else:
            schema_query = self.schema_store.table.search()
            value_query = self.value_store.table.search()

        # Search schemas
        schema_results = (
            schema_query.select(
                ["identifier", "catalog_name", "database_name", "schema_name", "table_name", "table_type", "definition"]
            )
            .limit(len(tables))
            .to_arrow()
        )
        schemas_result = TableSchema.from_arrow(schema_results)

            value_results = (
                value_query.select(
                    [
                        "identifier",
                        "catalog_name",
                        "database_name",
                        "schema_name",
                        "table_name",
                        "table_type",
                        "column_name",
                        "sample_rows",
                    ]
                )
                .limit(len(tables))
                .to_arrow()
        )
        values_result = TableValue.from_arrow(value_results)

        return schemas_result, values_result

    @staticmethod
    def _build_column_type_map(schemas: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, str]]:
        column_type_map: Dict[str, Dict[str, str]] = {}
        if not schemas:
            return column_type_map

        for schema in schemas:
            identifier = schema.get("identifier", "")
            definition = schema.get("definition", "")
            if not identifier or not definition:
                continue

            column_types = SchemaWithValueRAG._parse_column_types_from_definition(definition)
            if column_types:
                column_type_map[identifier] = column_types

        return column_type_map

    @staticmethod
    def _parse_column_types_from_definition(definition: str) -> Dict[str, str]:
        match = re.search(r"\((.*)\)", definition, re.DOTALL)
        if not match:
            return {}

        columns_block = match.group(1)
        column_defs = [segment.strip() for segment in re.split(r",\s*(?![^()]*\))", columns_block)]
        column_types: Dict[str, str] = {}
        for col_def in column_defs:
            if not col_def:
                continue
            upper_def = col_def.upper()
            if upper_def.startswith(("PRIMARY KEY", "UNIQUE", "CONSTRAINT", "FOREIGN", "KEY", "INDEX")):
                continue
            parts = col_def.split()
            if len(parts) < 2:
                continue
            column_name = parts[0].strip('"`[]')
            column_type = parts[1]
            column_types[column_name] = column_type
        return column_types

    @staticmethod
    def _is_textual_type(column_type: Optional[str]) -> bool:
        if not column_type:
            return True
        upper_type = column_type.upper()
        text_keywords = ["CHAR", "VARCHAR", "TEXT", "STRING", "NCHAR", "NVARCHAR", "CLOB", "JSON"]
        return any(keyword in upper_type for keyword in text_keywords)

    @staticmethod
    def _truncate_value(value: str, max_length: int = 512) -> str:
        if len(value) <= max_length:
            return value
        return value[:max_length] + "...(truncated)"

    @staticmethod
    def _expand_sample_rows_by_column(
        item: Dict[str, Any], column_type_map: Optional[Dict[str, Dict[str, str]]] = None
    ) -> List[Dict[str, Any]]:
        column_records: List[Dict[str, Any]] = []
        sample_rows = item.get("sample_rows")
        if not sample_rows:
            return column_records

        reader = csv.DictReader(StringIO(sample_rows))
        if not reader.fieldnames:
            return column_records

        column_types = (column_type_map or {}).get(item.get("identifier", ""), {})

        column_values: Dict[str, List[str]] = {field: [] for field in reader.fieldnames}
        for row in reader:
            for column_name, value in row.items():
                column_values[column_name].append("" if value is None else SchemaWithValueRAG._truncate_value(str(value)))

        base_fields = {
            "identifier": item.get("identifier", ""),
            "catalog_name": item.get("catalog_name", ""),
            "database_name": item.get("database_name", ""),
            "schema_name": item.get("schema_name", ""),
            "table_name": item.get("table_name", ""),
            "table_type": item.get("table_type", "table"),
        }

        for column_name, values in column_values.items():
            if not values:
                continue

            if not SchemaWithValueRAG._is_textual_type(column_types.get(column_name)):
                continue

            column_records.append(
                {
                    **base_fields,
                    "column_name": column_name,
                    "sample_rows": f"{column_name}:{' | '.join(values)}",
                }
            )

        return column_records

    def remove_data(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_name: str = "",
        table_type: TABLE_TYPE = "table",
    ):
        # Ensure tables are ready before deletion
        self.schema_store._ensure_table_ready()
        self.value_store._ensure_table_ready()

        where_condition = _build_where_clause(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_name=table_name,
            table_type=table_type,
        )
        where_clause = build_where(where_condition) if where_condition else None
        self.schema_store.table.delete(where_clause)
        self.value_store.table.delete(where_clause)


def _build_where_clause(
    catalog_name: str = "",
    database_name: str = "",
    schema_name: str = "",
    table_name: str = "",
    table_type: TABLE_TYPE = "table",
) -> Optional[Node]:
    conditions = []
    if catalog_name:
        conditions.append(eq("catalog_name", catalog_name))
    if database_name:
        conditions.append(eq("database_name", database_name))
    if schema_name:
        conditions.append(eq("schema_name", schema_name))
    if table_name:
        conditions.append(eq("table_name", table_name))
    if table_type and table_type != "full":
        conditions.append(eq("table_type", table_type))

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return and_(*conditions)
