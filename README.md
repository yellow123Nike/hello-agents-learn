# hello-agents-learn
《从零开始构建智能体》——从零开始的智能体原理与实践教程 学习

# agent架构
    与HelloAgents保持一致，agent架构分为三层：
    核心架构层(core):包含agent基类、llm调用、消息管理、异常管理、配置管理
    agent范式(agents_design):包含几种常见的agent范式-react、reflection、plan-solve
    工具层(tool):rag、mcp、skill等一概统一为工具

# core

## llm
1.支持多提供商
    hello_agent:不同模型服务提供商在模型调用的入参结构、返回格式等方面存在差异。为实现统一接入与平滑切换，引入 Provider 抽象层，在 LLM 调用内部对各服务商的协议差异、参数映射与配置细节进行封装与适配，从而对上层业务提供一致、稳定的模型调用接口。实际实现是仅支持openai-style接口，对于其它接口类型如claude等并不支持。
    our：LLMParams负责语义参数、provider负责把语义参数映射为具体API参数，以此实现支持多提供商。如gpt4-max_token和gpt5-max_completion_tokens

## message
1.一个清晰的上下文管理对模型调用很重要
