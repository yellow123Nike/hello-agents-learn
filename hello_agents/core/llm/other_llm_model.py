import asyncio
from openai import AsyncOpenAI
from typing import List, Dict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import RateLimitError, APIConnectionError, Timeout

from hello_agents.core.llm.llm_schema import PROVIDERS, LLMParams


class LLMClient_OpenAI:
    def __init__(self, params: LLMParams, provider_name: str):
        self.params = params
        self.provider = PROVIDERS[provider_name]
        self.client = AsyncOpenAI(
            api_key=params.api_key,
            base_url=params.base_url,
        )

    # messages 构建
    def build_messages(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        history_messages: List[Dict] | None = None,
    ) -> List[Dict]:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history_messages:
            messages.extend(history_messages)

        messages.append({"role": "user", "content": user_prompt})
        return messages

    def build_request_params(self, messages, stream, tools=None, tool_choice=None):
        payload = {
            "model": self.params.model_name,
            "messages": messages,
            "temperature": self.params.temperature,
            "stream": stream,
        }

        payload[self.provider.max_tokens_field] = self.params.max_tokens
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        return payload

    @retry(
        stop=stop_after_attempt(3),                 # 最多重试 3 次
        wait=wait_exponential(multiplier=1, min=2, max=10),  # 指数退避
        retry=retry_if_exception_type(
            (
                RateLimitError,
                APIConnectionError,
                Timeout,
                asyncio.TimeoutError,
            )
        ),
        reraise=True,   # 最终失败时抛出原异常
    )
    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: List[Dict] | None = None,
    ) -> str:
        # 1.messages组装
        messages = self.build_messages(prompt, system_prompt, history_messages)
        payload = self.build_request_params(
            messages=messages, stream=False,
        )
        response = await asyncio.wait_for(self.client.chat.completions.create(
            **payload
        ), timeout=240)

        return response.choices[0].message.content


    async def embedding_complete(self, inputs, dimensions=1024):
        BATCH_SIZE = 32
        all_embeddings = []

        for i in range(0, len(inputs), BATCH_SIZE):
            batch = inputs[i:i + BATCH_SIZE]

            response = await self.client.embeddings.create(
                input=batch,
                model=self.params.model_name,
                dimensions=dimensions
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings


# -------------------------------------------------------------------------
# ChromaDB 兼容的 Embedding 封装（vLLM / 类 OpenAI 接口）
# -------------------------------------------------------------------------


class VLLMChromaEmbeddingFunction:
    """
    将本地 vLLM（类 OpenAI）的 embedding 接口封装为 ChromaDB 可用的 embedding_function。
    ChromaDB 会同步调用 __call__(input: List[str])，内部用 asyncio.run 调用 LLMClient_OpenAI.embedding_complete。
    """

    def __init__(self, llm_client: LLMClient_OpenAI, dimensions: int = 1024):
        """
        Args:
            llm_client: 已配置好的 LLMClient_OpenAI（base_url 指向 vLLM embedding 服务）
            dimensions: 向量维度，与 vLLM 侧一致
        """
        self.llm_client = llm_client
        self.dimensions = dimensions

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        ChromaDB 调用的接口：input 为文本列表，返回等长的 embedding 列表。
        """
        if not input:
            return []
        return asyncio.run(self.llm_client.embedding_complete(input, dimensions=self.dimensions))

