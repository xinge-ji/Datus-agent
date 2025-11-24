# Modeling Rules Configuration

> Persist SQL modeling standards so the agent can learn them once and reuse them across sessions and projects.

## Purpose

Use the `modeling` section of `agent.yml` to capture the conventions that shape your semantic layer or transformation projects. Datus keeps these rules alongside other agent settings, enabling the LLM to recall how you name models, map layers, and enforce retention policies without re-prompting every session.

## Configuration

```yaml
modeling:
  naming_conventions:
    tables: snake_case
    dimensions: dim_<subject>
    facts: fct_<subject>
  layer_mapping:
    raw: bronze
    cleaned: silver
    marts: gold
  retention:
    staging: 7d
    marts: 180d
  notes:
    - prefer incremental materializations for wide fact tables
    - keep date dimensions shared across layers
```

### Fields

- **naming_conventions**: Key/value pairs describing how model types should be named (e.g., `dimensions`, `facts`, or custom categories). The agent can reuse these rules when proposing new models.
- **layer_mapping**: Maps layer names from your project to standardized terminology so the agent aligns recommendations with your stack (e.g., raw → bronze → silver → gold).
- **retention**: Guidance on how long each layer should persist. This helps the agent prefer transient staging objects or durable marts appropriately.
- **notes**: Free-form reminders the agent should keep in mind when generating or refactoring models (e.g., incremental preferences, shared dimensions).

## Best Practices

- Keep naming patterns succinct—use `<subject>` placeholders the agent can substitute.
- Align `layer_mapping` keys with the terms used in your transformation tool (e.g., SQLMesh or dbt environment names).
- Record retention in human-friendly units (e.g., `7d`, `30d`, `6mo`); the agent will surface this guidance in explanations.
- Update the section whenever conventions change to keep future generations consistent.
