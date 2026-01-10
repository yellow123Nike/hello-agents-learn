SENSITIVE_PATTERNS = {
    "id_card": r"\b\d{17}[\dXx]\b",
}

STRUCT_PARSE_TOOL_SYSTEM_PROMPT = """
## 工具 - Tools
### 输出工具的格式 - Tool Format
- 请结合前面的要求，严格输出JSON格式内容
- 文字内容提及需要使用工具列表中的工具时，在最后输出对应工具名的JSON格式内容
- 工具调用时，输出单个工具调用的JSON格式，格式示例如下：
```json
    {"function_name": "工具名1", "query": "xxx"}
```
- 工具调用时，输出多个不同工具调用的JSON格式，格式示例如下：
```json
{"function_name": "tool_a", "query": "xxx"}
```
```json
{"function_name": "tool_b", "query": "yyy"}
```
- 请理解上述JSON格式定义，仅输出最终的JSON格式。
- 输出的JSON的内容用双引号("")，不要用单引号('''')，并注意转义字符的使用
### 示例
    可用工具示例如下：
    - `deep_search`
      ```json
      {''name'': ''deep_search'', ''description'': ''这是一个搜索工具，可以搜索各种互联网知识'', ''parameters'': {''type'': ''object'', ''properties'': {''query'': {''description'': ''需要搜索的全部内容及描述'', ''type'': ''string''}}, ''required'': [''query'']}}
      ``` 
    工具调用输出的示例格式如下：
       ```json
      {"function_name": "deep_search", "query": "xxx"}
      ```    
### 约束
      - 先输出文字内容，再输出工具调用的JSON格式
      - 你只能能输出工具列表中的一个或多个，严禁输出工具列表中不存在的工具名
      - 不要自行补充或者臆造内容
      - 禁止输出多个相同入参的工具调用
 ### 工具列表 - Tool
      有如下工具名和工具入参的介绍如下：
"""
