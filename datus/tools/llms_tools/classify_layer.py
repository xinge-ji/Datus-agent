import json
from typing import Any, Dict
from datus.models.base import LLMBaseModel
from datus.prompts.layer_classification import get_layer_classification_prompt
from datus.utils.loggings import get_logger
from datus.utils.json_utils import llm_result2json

logger = get_logger(__name__)


def classify_view_layer(
    model: LLMBaseModel, view_name: str, feature: Dict[str, Any], dependencies: list, ddl_sql: str
) -> Dict[str, Any]:
    """
    使用 LLM 分析视图层级
    """
    prompt = get_layer_classification_prompt(
        view_name=view_name, feature=feature, dependencies=dependencies, ddl_sql=ddl_sql
    )

    try:
        # 使用 json 模式生成
        response = model.generate_with_json_output(prompt)
        # 结果通常已经是 dict，如果模型实现不同可能需要解析
        if isinstance(response, str):
            response = llm_result2json(response)

        return {
            "layer": response.get("layer", "OTHER"),
            "description": response.get("description", ""),
            "confidence": response.get("confidence", 0.5),
        }
    except Exception as e:
        logger.error(f"Layer classification failed for {view_name}: {e}")
        return {"layer": "OTHER", "description": "Analysis failed", "confidence": 0.0}
