# MetricFlow ↔ SQLMesh Metric Synchronization

This guide explains how to keep MetricFlow semantic/metric definitions and SQLMesh metric models aligned so the agent can apply data warehouse best practices while keeping its knowledge base current.

## Conversion pipeline

1. **Author or update MetricFlow semantics/metrics** in the MetricFlow project (generated YAML or edits from the agent/user).
2. **Translate MetricFlow metrics into SQLMesh metric models**:
   - Use the same metric identifiers, dimensions, and filters so business meaning is preserved.
   - Map MetricFlow measure proxies or expressions to SQLMesh `metric`/`model` files (for example, `metrics/<metric_name>.sql` or `.yaml`) with equivalent aggregations and constraints.
   - Carry forward locked business metadata as comments or tags in SQLMesh to keep parity for governance.
3. **Validate both sides**:
   - Run MetricFlow validation (`mf validate-configs`) after MetricFlow changes.
   - Run SQLMesh `plan`/`apply` (or the project’s preferred deployment routine) to ensure translated metrics compile and deploy.

## Keeping SQLMesh up to date when MetricFlow changes

- When the agent generates or edits MetricFlow metrics, it must **regenerate the SQLMesh metric artifacts** so downstream SQLMesh projects stay in sync.
- Store the SQLMesh output in the project’s SQLMesh directory (for example, `sqlmesh/models/metrics/`) and re-run the project’s SQLMesh deployment commands.
- Record the sync in the agent’s knowledge base metadata so future prompts know the SQLMesh artifacts are authoritative for deployment.

## Learning from user edits inside the SQLMesh project

Users can edit SQLMesh projects directly. To keep the agent’s knowledge accurate:

1. **Monitor for changes**: Watch the SQLMesh project directory (git status, file watchers, or MCP filesystem reads) for updated metric files.
2. **Re-ingest definitions**: Parse the changed SQLMesh metric/model files and update the agent’s external knowledge store with the refreshed definitions, tags, and constraints.
3. **Propagate back to MetricFlow** (if MetricFlow is the canonical layer): Regenerate MetricFlow metric YAML from the SQLMesh updates so both layers stay aligned.
4. **Validate both layers** again to guarantee parity after ingesting user edits.

By following this loop, MetricFlow remains the modeling source of truth, SQLMesh stays deployment-ready, and the agent’s knowledge base reflects the latest project state even when users make direct SQLMesh modifications.
