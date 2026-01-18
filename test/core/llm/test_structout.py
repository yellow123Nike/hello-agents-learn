"""
结构化输出方法对比测试
测试不同结构化输出方法对结构化输出的稳定性、输出结果、推理速度、token消耗的影响
实现的方法：
 见微信公众号文章：https://mp.weixin.qq.com/s/PvUbufb-2Sw1yH9S1K0GDg
"""
import asyncio
import json
import re
import time
from typing import Dict, Any, List, Union, Type
from pydantic import BaseModel, ValidationError, Field
from openai import AsyncOpenAI
from hello_agents.core.llm.llm_schema import LLMParams, PROVIDERS
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import RateLimitError, APIConnectionError, Timeout


params = LLMParams(
    model_name="Qwen/Qwen3-32B-AWQ",  # 或使用其他模型
    api_key="sk-",
    base_url="http://0.0.0.0:1111/v1/",
    temperature=0,
    max_tokens=14000,
)

# 测试用例：GSM8K中文数据集示例
# 格式：question_zh, answer_zh, answer_only
# 读取 data/gsm8k_zh.json 文件，获取question_zh, answer_zh, answer_only
with open("data/gsm8k_zh.json", "r", encoding="utf-8") as f:
    data = json.load(f)


# Pydantic模型定义，用于数学问题求解的结构化输出
class Step(BaseModel):
    """解题步骤"""
    explanation: str = Field(description="步骤说明")
    output: str = Field(description="该步骤的输出结果")


class MathResponse(BaseModel):
    """数学问题求解结果模型"""
    steps: List[Step] = Field(description="解题步骤列表", min_items=1)
    final_answer: int = Field(description="最终答案")


# 保留JSON Schema定义用于API调用（如json_schema方法）
EXPECTED_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "output": {"type": "string"}
                },
                "required": ["explanation", "output"]
            }
        },
        "final_answer": {"type": "integer"}
    },
    "required": ["steps", "final_answer"]
}


class StructuredOutputTester:
    """结构化输出测试类"""

    def __init__(self, params: LLMParams, provider_name: str = "openai"):
        self.params = params
        self.provider = PROVIDERS[provider_name]
        self.client = AsyncOpenAI(
            api_key=params.api_key,
            base_url=params.base_url,
        )

    def build_messages(self, user_prompt: str, system_prompt: str | None = None) -> List[Dict]:
        """构建消息列表"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, Timeout)),
        reraise=True,
    )
    async def call_api(self, payload: Dict) -> Dict[str, Any]:
        """
        调用OpenAI API

        如果payload中的response_format是Pydantic模型类，使用chat.completions.parse()
        否则使用chat.completions.create()
        """
        start_time = time.time()

        # 检查response_format是否是Pydantic模型类
        response_format = payload.get("response_format")
        use_parse = (
            isinstance(response_format, type)
            and issubclass(response_format, BaseModel)
        )

        if use_parse:
            # 使用parse方法，直接返回Pydantic模型实例
            response = await asyncio.wait_for(
                self.client.chat.completions.parse(**payload),
                timeout=240
            )
            # parse()返回的response中，parsed是Pydantic模型实例
            parsed_model = response.choices[0].message.parsed
            # 将Pydantic模型转换为JSON字符串，保持与其他方法的一致性
            content = json.dumps(parsed_model.model_dump(
            ), ensure_ascii=False) if parsed_model else None
        else:
            # 使用create方法
            response = await asyncio.wait_for(
                self.client.chat.completions.create(**payload),
                timeout=240
            )
            content = response.choices[0].message.content

        duration = time.time() - start_time

        result = {
            "content": content,
            "duration": duration,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            "finish_reason": response.choices[0].finish_reason,
        }

        # 如果是function call，提取工具调用信息（只有create方法可能有tool_calls）
        if not use_parse and response.choices[0].message.tool_calls:
            result["tool_calls"] = [
                {
                    "function": call.function.name,
                    "arguments": call.function.arguments,
                }
                for call in response.choices[0].message.tool_calls
            ]

        return result

    async def test_prompt_engineering(self, user_prompt: str) -> Dict[str, Any]:
        """方法1: Prompt工程 - 使用精心设计的prompt和few-shot示例"""
        system_prompt = """你是一个信息提取助手。请严格按照JSON格式输出结果。"""

        messages = self.build_messages(user_prompt, system_prompt)
        payload = {
            "model": self.params.model_name,
            "messages": messages,
            "temperature": self.params.temperature,
            self.provider.max_tokens_field: self.params.max_tokens,
        }

        return await self.call_api(payload)

    async def test_json_mode(self, user_prompt: str) -> Dict[str, Any]:
        """方法2: JSON Mode - 使用response_format={"type": "json_object"}"""
        system_prompt = """你是一个信息提取助手。请提取文本中的人物信息。"""

        messages = self.build_messages(user_prompt, system_prompt)
        payload = {
            "model": self.params.model_name,
            "messages": messages,
            "temperature": self.params.temperature,
            "response_format": {"type": "json_object"},
            self.provider.max_tokens_field: self.params.max_tokens,
        }

        return await self.call_api(payload)

    async def test_json_schema(self, user_prompt: str, schema: Union[Type[BaseModel], Dict]) -> Dict[str, Any]:
        """
        方法3: JSON Schema - 使用response_format={"type": "json_schema", "json_schema": {...}}

        支持两种方式：
        1. 传入Pydantic模型类：自动转换为JSON Schema（推荐，类型安全）
        2. 传入JSON Schema字典：直接使用

        注意：此方法需要支持JSON Schema功能的模型（如gpt-4o, gpt-4-turbo等较新模型）
        如果模型不支持，API会返回错误
        """
        # 如果传入的是Pydantic模型类，转换为JSON Schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            json_schema_dict = schema.model_json_schema()
            schema_name = schema.__name__.lower()
        elif isinstance(schema, dict):
            json_schema_dict = schema
            schema_name = "person_extraction"  # 默认名称
        else:
            raise ValueError(f"schema必须是Pydantic模型类或字典，当前类型: {type(schema)}")

        system_prompt = """你是一个信息提取助手。请按照指定的JSON Schema格式输出结果。"""

        messages = self.build_messages(user_prompt, system_prompt)
        payload = {
            "model": self.params.model_name,
            "messages": messages,
            "temperature": self.params.temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": json_schema_dict,
                }
            },
            self.provider.max_tokens_field: self.params.max_tokens,
        }

        try:
            return await self.call_api(payload)
        except Exception as e:
            # 如果模型不支持json_schema格式，返回错误信息
            error_msg = str(e)
            if "json_schema" in error_msg.lower() or "response_format" in error_msg.lower():
                return {
                    "content": None,
                    "error": f"模型可能不支持JSON Schema格式: {error_msg}",
                    "duration": 0,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
            raise

    async def test_pydantic_schema(self, user_prompt: str, schema: Union[Type[BaseModel], Dict]) -> Dict[str, Any]:
        """
        方法3: Pydantic Schema - 使用Pydantic模型类，自动转换为JSON Schema

        支持两种方式：
        1. 传入Pydantic模型类（如PersonExtraction）：直接使用
        2. 传入JSON Schema字典：自动映射到对应的Pydantic模型类（如PersonExtraction）
        """
        # 如果传入的是Pydantic模型类，直接使用
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            pydantic_model = schema
            json_schema_dict = schema.model_json_schema()
            schema_name = schema.__name__.lower()
        elif isinstance(schema, dict):
            # 如果传入的是JSON Schema字典，映射到对应的Pydantic模型类
            # 这里可以通过比较schema结构来判断应该使用哪个Pydantic类
            # 对于当前场景，EXPECTED_SCHEMA对应PersonExtraction
            # 可以扩展为更通用的映射逻辑
            pydantic_model = MathResponse  # 将JSON Schema映射到MathResponse
            json_schema_dict = schema
            schema_name = "person_extraction"
        else:
            raise ValueError(f"schema必须是Pydantic模型类或字典，当前类型: {type(schema)}")

        system_prompt = """你是一个信息提取助手。请按照指定的JSON Schema格式输出结果。"""

        messages = self.build_messages(user_prompt, system_prompt)
        payload = {
            "model": self.params.model_name,
            "messages": messages,
            "temperature": self.params.temperature,
            "response_format": pydantic_model,
            self.provider.max_tokens_field: self.params.max_tokens,
        }

        try:
            return await self.call_api(payload)
        except Exception as e:
            # 如果模型不支持json_schema格式，返回错误信息
            error_msg = str(e)
            if "json_schema" in error_msg.lower() or "response_format" in error_msg.lower():
                return {
                    "content": None,
                    "error": f"模型可能不支持JSON Schema格式: {error_msg}",
                    "duration": 0,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
            raise

    async def test_function_call(self, user_prompt: str, schema: Dict) -> Dict[str, Any]:
        """方法4: Function Call - 使用tools参数进行工具调用"""
        system_prompt = """你是一个数学问题求解助手。请使用提供的工具来解决数学问题。"""

        # 将JSON Schema转换为OpenAI Function格式
        def schema_to_function(schema: Dict) -> Dict:
            """将JSON Schema转换为OpenAI Function格式"""
            # 直接使用传入的schema作为parameters
            return {
                "type": "function",
                "function": {
                    "name": "solve_math_problem",
                    "description": "解决数学问题并输出解题步骤和最终答案",
                    "parameters": schema
                }
            }

        function_def = schema_to_function(schema)

        messages = self.build_messages(user_prompt, system_prompt)
        payload = {
            "model": self.params.model_name,
            "messages": messages,
            "temperature": self.params.temperature,
            "tools": [function_def],
            "tool_choice": {"type": "function", "function": {"name": "solve_math_problem"}},
            self.provider.max_tokens_field: self.params.max_tokens,
        }

        result = await self.call_api(payload)

        # 从tool_calls中提取JSON内容
        if "tool_calls" in result and result["tool_calls"]:
            result["content"] = result["tool_calls"][0]["arguments"]

        return result

    def validate_output(self, content: str, method_name: str, answer_only: str = None) -> Dict[str, Any]:
        """
        使用Pydantic验证输出结果的正确性

        Args:
            content: 模型输出的JSON字符串
            method_name: 方法名称（用于记录）
            answer_only: 标准答案（用于验证final_answer是否正确）

        Returns:
            验证结果字典，包含：
            - is_valid_json: JSON是否有效
            - matches_schema: 是否符合Pydantic模型定义的schema
            - answer_correct: final_answer是否与answer_only匹配
            - error: 错误信息（如果有）
            - parsed_content: 解析后的内容
            - validation_errors: Pydantic验证错误详情（如果有）
        """
        validation_result = {
            "is_valid_json": False,
            "matches_schema": False,
            "answer_correct": False,
            "error": None,
            "parsed_content": None,
            "validation_errors": None,
        }

        try:
            # 如果是prompt工程，则对结果进行正则提取
            if method_name == "prompt_engineering":
                # 先尝试提取 ```json ... ``` 格式
                pattern1 = r"```json\s*([\s\S]*?)\s*```"
                match = re.search(pattern1, content)
                if match:
                    content = match.group(1).strip()
                else:
                    # 如果没有代码块，尝试提取JSON对象（处理<think>等标记）
                    # 先移除各种XML/HTML标签：<think>...</think>等
                    content = re.sub(r'<[^>]+>[\s\S]*?</[^>]+>', '', content)
                    # 移除单独的标签如<think>或</think>
                    content = re.sub(r'<[^>]+>', '', content)
                    # 提取JSON对象 { ... }，使用括号匹配找到完整的JSON对象
                    start_idx = content.find('{')
                    if start_idx != -1:
                        # 从第一个{开始，使用括号匹配找到匹配的最后一个}
                        brace_count = 0
                        end_idx = start_idx
                        for i in range(start_idx, len(content)):
                            if content[i] == '{':
                                brace_count += 1
                            elif content[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_idx = i
                                    break
                        if end_idx > start_idx:
                            content = content[start_idx:end_idx+1].strip()

                # 尝试解析提取的JSON
                if content:
                    parsed = json.loads(content)
                    validation_result["is_valid_json"] = True
                    validation_result["parsed_content"] = parsed
                else:
                    validation_result["error"] = "无法从输出中提取JSON内容"
                    return validation_result
            else:
                # 尝试解析JSON
                parsed = json.loads(content)
                validation_result["is_valid_json"] = True
                validation_result["parsed_content"] = parsed

            # 使用Pydantic验证是否符合schema
            try:
                # Pydantic会自动进行类型检查和验证
                validated_data = MathResponse.model_validate(parsed)
                validation_result["matches_schema"] = True
                # 将Pydantic模型转换回字典，确保类型一致
                validation_result["parsed_content"] = validated_data.model_dump()

                # 验证final_answer是否与answer_only匹配
                if answer_only is not None:
                    try:
                        # 将answer_only转换为数值进行比较
                        expected_answer = float(answer_only.strip())
                        actual_answer = float(validated_data.final_answer)
                        # 使用小的容差进行比较（避免浮点数精度问题）
                        validation_result["answer_correct"] = abs(
                            expected_answer - actual_answer) < 0.001
                    except (ValueError, TypeError):
                        # 如果转换失败，尝试字符串比较
                        validation_result["answer_correct"] = str(
                            validated_data.final_answer).strip() == str(answer_only).strip()

            except ValidationError as e:
                # Pydantic验证失败，记录详细错误信息
                validation_result["matches_schema"] = False
                validation_result["validation_errors"] = e.errors()
                validation_result["error"] = f"Pydantic验证失败: {str(e)}"

        except json.JSONDecodeError as e:
            validation_result["error"] = f"JSON解析错误: {str(e)}"
        except Exception as e:
            validation_result["error"] = f"验证错误: {str(e)}"

        return validation_result

    def _build_user_prompt_with_schema(self, prompt: str) -> str:
        """
        构建包含格式说明和示例的user_prompt（数学问题求解场景）

        Args:
            prompt: 原始输入文本（数学问题）

        Returns:
            包含格式说明和示例的完整user_prompt
        """
        # 创建一个示例数据用于展示格式
        example_data = MathResponse(
            steps=[
                Step(explanation="计算苹果的总价", output="3 × 2 = 6"),
                Step(explanation="计算橙子的总价", output="5 × 1.5 = 7.5"),
                Step(explanation="计算总价", output="6 + 7.5 = 13.5")
            ],
            final_answer=13  # 注意：Pydantic模型中final_answer是int类型，实际答案应该是13或14（四舍五入）
        )
        example_json = json.dumps(
            example_data.model_dump(), ensure_ascii=False, indent=2)

        user_prompt = f"""请解决以下数学问题，并按照指定格式输出解题步骤和最终答案。

            输出格式要求：
            - 必须是一个有效的JSON对象
            - 包含一个"steps"数组，数组中的每个对象包含以下字段：
            * explanation: 字符串类型，表示该步骤的说明
            * output: 字符串类型，表示该步骤的计算结果
            - 包含一个"final_answer"字段：整数类型，表示最终答案

            示例格式：
            {example_json}

            请解决以下问题：
            {prompt}
            /no_think
        """

        return user_prompt

    async def run_comparison(self, prompt: str, schema: Union[Type[BaseModel], Dict], answer_only: str = None, runs: int = 1) -> Dict[str, Any]:
        """
        运行所有方法的对比测试

        Args:
            prompt: 原始输入文本（数学问题）
            schema: Pydantic模型类或JSON Schema字典（用于json_schema和function_call方法）
            answer_only: 标准答案（用于验证final_answer是否正确）
            runs: 每种方法运行的次数，用于计算平均值
        """
        # 在run_comparison中统一构建user_prompt，确保所有方法使用完全相同的输入
        # 包含格式说明和示例，这样json_mode等方法也能生成符合格式的输出
        user_prompt = self._build_user_prompt_with_schema(prompt)

        results = {
            "prompt": prompt,
            "user_prompt": user_prompt,
            "methods": {},
            "summary": {}
        }

        methods = {
            # "prompt_engineering": self.test_prompt_engineering,
            "json_mode": self.test_json_mode,
            # "json_schema": self.test_json_schema,
            # "function_call": self.test_function_call,
            # "pydantic_schema": self.test_pydantic_schema,
        }

        # 运行每种方法
        for method_name, method_func in methods.items():
            print(f"\n{'='*60}")
            print(f"测试方法: {method_name}")
            print(f"{'='*60}")

            method_results = []

            for run_idx in range(runs):
                print(f"运行 {run_idx + 1}/{runs}...")

                try:
                    # 根据方法类型调用不同的函数，所有方法都使用统一的user_prompt
                    if method_name == "json_schema" or method_name == "function_call" or method_name == "pydantic_schema":
                        result = await method_func(user_prompt, schema)
                    else:
                        result = await method_func(user_prompt)

                    # 检查是否有错误
                    if "error" in result:
                        print(f"  错误: {result['error']}")
                        result["validation"] = {
                            "is_valid_json": False,
                            "matches_schema": False,
                            "error": result['error'],
                            "parsed_content": None,
                        }
                        method_results.append(result)
                    else:
                        # 验证输出（传入answer_only用于答案验证）
                        validation = self.validate_output(result.get(
                            "content") or "", method_name, answer_only=answer_only)
                        result["validation"] = validation

                        method_results.append(result)

                        # 打印单次结果
                        print(f"  响应时间: {result.get('duration', 0):.2f}秒")
                        print(
                            f"  输入tokens: {result.get('usage', {}).get('prompt_tokens', 0)}")
                        print(
                            f"  输出tokens: {result.get('usage', {}).get('completion_tokens', 0)}")
                        print(
                            f"  总tokens: {result.get('usage', {}).get('total_tokens', 0)}")
                        if validation:
                            print(
                                f"  有效JSON: {validation.get('is_valid_json', False)}")
                            print(
                                f"  符合Schema: {validation.get('matches_schema', False)}")
                            if answer_only is not None:
                                print(
                                    f"  答案正确: {validation.get('answer_correct', False)}")

                except Exception as e:
                    print(f"  错误: {str(e)}")
                    method_results.append({
                        "error": str(e),
                        "duration": 0,
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    })

            # 计算平均值
            successful_runs = [r for r in method_results if "error" not in r]
            if method_results and len(successful_runs) > 0:
                avg_result = {
                    "avg_duration": sum(r.get("duration", 0) for r in successful_runs) / len(successful_runs),
                    "avg_prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in successful_runs) / len(successful_runs),
                    "avg_completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in successful_runs) / len(successful_runs),
                    "avg_total_tokens": sum(r.get("usage", {}).get("total_tokens", 0) for r in successful_runs) / len(successful_runs),
                    "valid_json_rate": sum(1 for r in successful_runs if r.get("validation", {}).get("is_valid_json", False)) / len(method_results),
                    "schema_match_rate": sum(1 for r in successful_runs if r.get("validation", {}).get("matches_schema", False)) / len(method_results),
                    "answer_correct_rate": sum(1 for r in successful_runs if r.get("validation", {}).get("answer_correct", False)) / len(method_results) if answer_only else None,
                    "success_rate": len(successful_runs) / len(method_results),
                    "all_runs": method_results,
                }
            else:
                avg_result = {
                    "error": "所有运行都失败",
                    "success_rate": 0,
                    "all_runs": method_results,
                }

            results["methods"][method_name] = avg_result

        # 生成总结
        results["summary"] = self._generate_summary(results["methods"])

        return results

    def _generate_summary(self, methods: Dict) -> Dict[str, Any]:
        """生成对比总结"""
        summary = {
            "fastest_method": None,
            "most_tokens_efficient": None,
            "most_accurate": None,
            "comparison_table": []
        }

        fastest_time = float('inf')
        least_tokens = float('inf')
        best_accuracy = 0

        for method_name, method_data in methods.items():
            if "error" not in method_data:
                # 速度对比
                if method_data["avg_duration"] < fastest_time:
                    fastest_time = method_data["avg_duration"]
                    summary["fastest_method"] = method_name

                # Token效率对比
                if method_data["avg_total_tokens"] < least_tokens:
                    least_tokens = method_data["avg_total_tokens"]
                    summary["most_tokens_efficient"] = method_name

                # 准确性对比
                accuracy = method_data["schema_match_rate"]
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    summary["most_accurate"] = method_name

                row = {
                    "method": method_name,
                    "avg_duration": round(method_data["avg_duration"], 3),
                    "avg_prompt_tokens": round(method_data["avg_prompt_tokens"], 0),
                    "avg_completion_tokens": round(method_data["avg_completion_tokens"], 0),
                    "avg_total_tokens": round(method_data["avg_total_tokens"], 0),
                    "valid_json_rate": round(method_data["valid_json_rate"], 2),
                    "schema_match_rate": round(method_data["schema_match_rate"], 2),
                }
                # 如果有answer_correct_rate，添加到表格中
                if method_data.get("answer_correct_rate") is not None:
                    row["answer_correct_rate"] = round(
                        method_data["answer_correct_rate"], 2)
                summary["comparison_table"].append(row)

        return summary


async def main():
    """主函数：运行对比测试（GSM8K数据集前1000个样本）"""
    tester = StructuredOutputTester(params, provider_name="openai")

    # 选择前1000个样本
    test_samples = data[:100]
    total_samples = len(test_samples)

    print("="*60)
    print("结构化输出方法对比测试 - GSM8K中文数据集")
    print("="*60)
    print(f"\n测试样本数: {total_samples}")
    print(f"运行次数: 每个样本运行1次")

    # 存储所有样本的结果
    all_results = []
    method_stats = {}  # 用于汇总统计

    # 定义单个样本的测试函数
    async def test_single_sample(idx: int, sample: Dict) -> Dict[str, Any]:
        """测试单个样本"""
        question_zh = sample.get("question_zh", "")
        answer_only = sample.get("answer_only", "")

        if not question_zh or not answer_only:
            return {
                "sample_idx": idx,
                "skipped": True,
                "reason": "缺少question_zh或answer_only"
            }

        try:
            # 运行对比测试
            # 方式1：使用JSON Schema字典（默认方式）
            results = await tester.run_comparison(
                question_zh,
                EXPECTED_SCHEMA,
                answer_only=answer_only,
                runs=1
            )
            # 方式2：使用Pydantic模型类（推荐，类型安全，会自动转换为JSON Schema）
            # results = await tester.run_comparison(question_zh, MathResponse, answer_only=answer_only, runs=1)

            return {
                "sample_idx": idx,
                "question_zh": question_zh,
                "answer_only": answer_only,
                "results": results,
                "skipped": False
            }
        except Exception as e:
            return {
                "sample_idx": idx,
                "skipped": True,
                "reason": f"测试失败: {str(e)}"
            }

    # 批量处理：每10个样本一批次并行运行
    batch_size = 10
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_samples = test_samples[batch_start:batch_end]
        batch_indices = list(range(batch_start + 1, batch_end + 1))

        print(f"\n{'='*60}")
        print(
            f"批次 {batch_start // batch_size + 1}/{(total_samples + batch_size - 1) // batch_size}")
        print(f"处理样本 {batch_start + 1}-{batch_end}/{total_samples}")
        print(f"{'='*60}")

        # 并行运行当前批次的所有样本
        batch_tasks = [
            test_single_sample(idx, sample)
            for idx, sample in zip(batch_indices, batch_samples)
        ]
        batch_results = await asyncio.gather(*batch_tasks)

        # 处理批次结果
        for result in batch_results:
            if result.get("skipped", False):
                print(
                    f"跳过样本 {result['sample_idx']}: {result.get('reason', '未知原因')}")
                continue

            all_results.append(result)

            # 汇总统计信息
            results = result.get("results", {})
            for method_name, method_data in results.get("methods", {}).items():
                if "error" not in method_data:
                    if method_name not in method_stats:
                        method_stats[method_name] = {
                            "total_samples": 0,
                            "total_duration": 0,
                            "total_prompt_tokens": 0,
                            "total_completion_tokens": 0,
                            "total_tokens": 0,
                            "valid_json_count": 0,
                            "schema_match_count": 0,
                            "answer_correct_count": 0,
                        }

                    stats = method_stats[method_name]
                    stats["total_samples"] += 1
                    stats["total_duration"] += method_data.get(
                        "avg_duration", 0)
                    stats["total_prompt_tokens"] += method_data.get(
                        "avg_prompt_tokens", 0)
                    stats["total_completion_tokens"] += method_data.get(
                        "avg_completion_tokens", 0)
                    stats["total_tokens"] += method_data.get(
                        "avg_total_tokens", 0)

                    # 检查单次运行的结果
                    for run in method_data.get("all_runs", []):
                        validation = run.get("validation", {})
                        if validation.get("is_valid_json", False):
                            stats["valid_json_count"] += 1
                        if validation.get("matches_schema", False):
                            stats["schema_match_count"] += 1
                        if validation.get("answer_correct", False):
                            stats["answer_correct_count"] += 1

        print(f"批次完成，已处理 {len(all_results)}/{total_samples} 个样本")

    # 计算总体统计
    print("\n" + "="*60)
    print("总体统计结果")
    print("="*60)

    if method_stats:
        print(f"\n测试样本数: {total_samples}")
        print(f"成功测试数: {len(all_results)}")

        print("\n对比表格（总体平均）:")
        print(f"{'方法':<20} {'平均耗时(秒)':<15} {'输入Tokens':<12} {'输出Tokens':<12} {'总Tokens':<12} {'JSON有效率':<12} {'Schema匹配率':<12} {'答案正确率':<12}")
        print("-" * 110)

        for method_name, stats in method_stats.items():
            if stats["total_samples"] > 0:
                avg_duration = stats["total_duration"] / stats["total_samples"]
                avg_prompt_tokens = stats["total_prompt_tokens"] / \
                    stats["total_samples"]
                avg_completion_tokens = stats["total_completion_tokens"] / \
                    stats["total_samples"]
                avg_total_tokens = stats["total_tokens"] / \
                    stats["total_samples"]
                valid_json_rate = stats["valid_json_count"] / \
                    stats["total_samples"] if stats["total_samples"] > 0 else 0
                schema_match_rate = stats["schema_match_count"] / \
                    stats["total_samples"] if stats["total_samples"] > 0 else 0
                answer_correct_rate = stats["answer_correct_count"] / \
                    stats["total_samples"] if stats["total_samples"] > 0 else 0

                print(f"{method_name:<20} {avg_duration:<15.3f} "
                      f"{avg_prompt_tokens:<12.0f} {avg_completion_tokens:<12.0f} {avg_total_tokens:<12.0f} "
                      f"{valid_json_rate:<12.2f} {schema_match_rate:<12.2f} {answer_correct_rate:<12.2f}")

    # 保存详细结果到文件（可选）
    # with open("test_results.json", "w", encoding="utf-8") as f:
    #     json.dump(all_results, f, indent=2, ensure_ascii=False)
    # print(f"\n详细结果已保存到 test_results.json")


if __name__ == "__main__":
    asyncio.run(main())
