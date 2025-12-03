from typing import Dict, List, Any
from .prompt_manager import prompt_manager


def get_layer_classification_prompt(
    view_name: str,
    feature: Dict[str, Any],
    dependencies: List[Dict[str, Any]],
    ddl_sql: str,
    prompt_version: str = "1.0",
) -> List[Dict[str, str]]:
    # 截取 DDL 避免 token 过长
    ddl_preview = ddl_sql[:2000] + "..." if len(ddl_sql) > 2000 else ddl_sql

    system_content = prompt_manager.get_raw_template("layer_classification_system", version=prompt_version)
    user_content = prompt_manager.render_template(
        "layer_classification_user",
        view_name=view_name,
        feature=feature,
        dependencies=dependencies,
        ddl_preview=ddl_preview,
        version=prompt_version,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
