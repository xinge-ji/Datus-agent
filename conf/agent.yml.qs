agent:
  target: deepseek-v3
  models:
    deepseek-v3:
      type: deepseek
      vendor: deepseek
      base_url: https://api.deepseek.com
      api_key: ${DEEPSEEK_API_KEY}
      model: deepseek-chat
    kimi-k2-turbo:
      type: openai
      vendor: openai
      base_url: https://api.moonshot.cn/v1
      api_key: ${KIMI_API_KEY}
      model: kimi-k2-turbo-preview

  storage:
    # Data path is now fixed at {agent.home}/data (e.g., ~/.datus/data/datus_db_{namespace})
    workspace_root: ~/.datus/workspace
    embedding_device_type: cpu
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
  benchmark:
    california_schools:
      question_file: california_schools.csv
      question_id_key: task_id
      question_key: question
      ext_knowledge_key: evidence
      gold_sql_path: california_schools.csv
      gold_sql_key: gold_sql
      gold_result_path: california_schools.csv

  namespace:
    local_duckdb:
      type: duckdb
      name: duckdb-demo
      uri: ~/.datus/sample/duckdb-demo.duckdb
    california_schools:
      type: sqlite
      name: california_schools
      uri: ~/.datus/benchmark/california_schools/california_schools.sqlite

  nodes:
    schema_linking:
      matching_rate: fast
    generate_sql:
      prompt_version: "1.0"
    reasoning:
      prompt_version: "1.0"
    reflect:
      prompt_version: "2.1"
    chat:
      prompt_version: "1.0"