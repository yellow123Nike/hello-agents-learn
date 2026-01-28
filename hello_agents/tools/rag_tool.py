from hello_agents.core.tool.base_tool import BaseTool


class RagTool(BaseTool):
    name = "rag"
    description = (
        "外部知识库，用于解决模型内置知识的局限(LLM 的知识是静态的、有限的)"
    )
