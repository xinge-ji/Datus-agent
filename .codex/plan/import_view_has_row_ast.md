任务：import_view 流程优化，视图 has_row 改用 AST 依赖判定并重排流水线提升性能。
上下文：
- 新流水线：先全 sourcedb 表导入（探活），再逐个 sourcedb 视图导入 + AST + has_row + feature；classify/naming 后置。
- 规则：AST 依赖判定 has_row：依赖表全 1 则 1；依赖视图/EXTERNAL/未知或 AST 失败默认 1；有表 has_row=0 则 0；has_row=0 时 parse_status=SKIPPED。
- 增量模式复用已有 has_row；feature 缓存命中且 source_hash 相同则复用，无需重跑 AST。
计划：
1) 扩展 table_source_map 支持 has_row，可用于依赖判定（已完成）。
2) 新增 _infer_has_row_from_ast（可复用缓存 feature），禁用虚拟外部，基于依赖判定 has_row（已完成）。
3) 重排 run(): import_tables -> import_views_with_ast；import_views 视图导入时写 feature，支持 feature_cache 命中复用，统计跳过/失败（已完成）。
状态：全部完成。
