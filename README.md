# hello-agents-learn
《从零开始构建智能体》——从零开始的智能体原理与实践教程 学习
具体实现大部分以joyagent-jdagenie逻辑实现

# agent架构
    与HelloAgents保持一致，agent架构分为三层：
    核心架构层(core):包含llm调用、agent基类、消息管理、异常管理、配置管理
    agent范式(agents_design):包含几种常见的agent范式-react、reflection、plan-solve
    工具层(tool):rag、mcp、skill等一概统一为工具

# core
## llm
1.支持多提供商
    hello_agent:不同模型服务提供商在模型调用的入参结构、返回格式等方面存在差异。为实现统一接入与平滑切换，引入 Provider 抽象层，在 LLM 调用内部对各服务商的协议差异、参数映射与配置细节进行封装与适配，从而对上层业务提供一致、稳定的模型调用接口。实际实现是仅支持openai-style接口，对于其它接口类型如claude等并不支持。
    our：LLMParams负责语义参数、provider负责把语义参数映射为具体API参数，以此实现支持多提供商。如gpt4-max_token和gpt5-max_completion_tokens。这里的设计没有采用多层继承实现，因为仅适配openai_style接口

2.上下文消息管理
    准确、稳定的上下文信息是保障大模型交互质量与推理一致性的关键。
    ·消息格式
        Openai规范中消息类型分为4种：
            ·user：用户输入消息
            ·system：系统级指令与约束信息
            ·assistant：模型生成的回复消息
            ·tool：工具执行结果消息
        和hello_agent一样，通过 Message 对象对各类消息进行统一建模，并在调用阶段将其转换为 OpenAI 兼容的字典结构。我们额外引入了对多模态内容的原生支持，可在同一消息中同时承载文本与图像等非结构化输入，为后续多模态推理与交互场景提供基础能力。
    ·上下文消息管理
        针对大模型存在的最大上下文长度限制，我额外加了一套可控、稳定的上下文管理机制，以保障模型调用阶段的鲁棒性。具体流程如下：
        ·消息格式化：
            首先对原始 message 列表进行统一格式化，生成符合模型接口规范的上下文消息序列。
        ·上下文长度评估
            在消息送入模型前，对当前上下文的 token 数进行评估，判断是否超过模型可接受的最大上下文限制。
        ·上下文截断策略
            当上下文长度超限时，系统采用倒序保留策略，优先保留最新的对话信息，并确保最后出现的非 system 消息为 user 类型，从而保证模型始终能够感知用户的最新问题与意图。
    ·上下文脱敏
        另外对执行结果进行了脱敏

3.模型调用
    我们提供了3个模型调用的原生接口：   
        ask_llm_once：一次性获取模型完整回复
        ask_llm_stream：实时获取模型生成结果
        ask_tool：工具/函数调用
    
## agent
    Agent抽象类(BaseAgent)：定义一个智能体应该具备的通用行为和属性，并不关心具体实现方式。
        __init__()清晰定义agent的核心依赖：
            name：agent名称、
            description：agent职责描述、
            system_prompt：我是谁、我能做什么、不该做什么
            next_step_prompt：这一轮我该做什么决策
            llm：模型API
            digital_employee_prompt：我在此agent场景下应该是一个什么角色
            available_tools：可用工具集合
            memory：上下文记忆
            context：agent基础配置
            state：agent状态(空闲\运行\完成\错误)
            max_steps:防止无限推理
            duplicate_threshold:防止模型陷入重复输出
        step():唯一抽象方法,这一轮 Agent 怎么想、怎么行动？
        run():生命周期控制+错误与终止处理
        execute_tools:agent的基础能力，和HelloAgents理念一致，一切皆工具
        update_memory：上下文内存管理,动态更新上下文
        generate_digital_employee:让 LLM 根据当前任务，动态“选择 / 生成一个合适的数字员工（角色 + 工具组合）”，并据此更新 Agent 当前可用的工具集合
    短期记忆组件(Memory): 负责维护、裁剪、读取“对话与执行历史”，为 LLM 下一步决策提供上下文。 

## printer
    输出方式抽象：把 Agent 的内部状态、过程和结果，以不同形式输出，Console/日志文件或其它方式

## tool
    工具抽象(basetool)：抽象工具描述(to_params)和工具执行逻辑(execute)。工具描述是告诉大模型这个工具需要什么参数。工具执行一旦 LLM 选定工具，系统就必须“无条件执行”
    mcptool：

# agents_design
    agents设计范式：react、reflection、plan-solve (hello-agent中的simpleagent本质上是一个react范式，故不实现，可直接通过react构造实例)

## react
    react范式的核心是边想边做，具有强环境适应性和动态纠错能力，是处理探索性、需要外部工具输入的任务的首选。
    react范式的局限性是步进式的决策模式意味着智能体缺乏一个全局的、长远的规划，可能陷入局部最优。
    具体范式可用抽象为：loop(思考-行动-观察)

    具体实现方式：
        严格拆分 Think 与 Act：
            Think → 决策下一步要做什么，调用 LLM 输出“下一步想法”或“要调用的工具”
            Act → 真正执行工具，把结果写入 memory
        Prompt 分层设计
            base_prompt → 用户指定 Agent 泛能力（业务/角色信息）
            system_prompt → ReAct 标准范式，指导 LLM 在react范式下如何思考
            next_step_prompt → 推进提示，保证 LLM 在非用户消息之后继续思考(这里有个小优化，如何防止next_step_prompt重复存储在memory中)
        Memory驱动
            所有 LLM 输出、工具调用、工具结果都写入 Memory，保证可追踪、可回溯

## plan-and-solve
    plan-and-solve的核心是想清楚再做，具有结构性和稳定性，适合处理逻辑路径确定、内部推理密集的任务。
    plan-and-solve的局限性是可能一次性想不清楚，没有每一步的结果很难想清楚
    具体范式可用抽象为：规划-loop(思考-行动-观察)
    具体实现方式1(is_close_update=True)：(这种方式在执行阶段可使用工具)
        规划阶段(think)：将问题分解，并制定出一个清晰、分步骤的行动计划。
        执行阶段(act-reactagent)：严格按照计划中的步骤，对于单一任务采用reactagent方式实现
    具体实现方式2(is_close_update=False)：(这种方式可动态更新任务)
        规划阶段(think)：将问题分解，并制定出一个清晰、分步骤的行动计划。+推进每一步的执行结果和状态
        执行阶段(act)：仅更新plantool的状态   
    plantool：管理和更新任务计划，包括生成、更新、执行和检查任务状态。它能够动态地推进每个步骤，并在必要时调整任务的顺序和状态。
    Planner：表示整个计划的内容和步骤，确保每个步骤都有明确的执行顺序和状态。负责任务的初始化、状态更新和进度管理。
    Prompt 分层设计：
        PLANSOLVE_SYSTEM_PROMPT：用于在规划阶段提供背景信息，包括任务的上下文、步骤以及必要的工具。此提示帮助规划代理理解任务的背景，并为生成计划提供支持
        PLANSOLVE_NEXT_STEP_PROMPT：用于在每个任务步骤执行时提供进一步的指引。这个提示可以动态变化，结合上一步执行的结果和任务的进度
        sop_prompt：通过明确任务的目标、步骤以及预期结果来指导代理
        base_prompt:reactagent指导