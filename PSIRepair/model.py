import os
import time
from functools import lru_cache
from typing import Optional, List, Dict, Union, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception,
)
from . import rs_utils as rsu
from .basic import Message, MessageList, LLMQueryRecord

try:
    import openai
except ImportError:
    openai = None


class ContextLengthExceededException(Exception):
    pass


_model_query_recorder = None


class LLMQueryRecorder:
    def __init__(self, **ext_info) -> None:
        self.records: List[LLMQueryRecord] = []
        self.__uni_ext_info = ext_info

    def __enter__(self):
        global _model_query_recorder
        self.__old_recorder = _model_query_recorder
        _model_query_recorder = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global _model_query_recorder
        _model_query_recorder = self.__old_recorder
        return False

    def append(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        start_at: float,
        finish_at: float,
        query: Union[str, MessageList],
        response: Union[str, Message],
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        **ext_info,
    ):
        self.records.append(
            LLMQueryRecord(
                id=f"{model_name}-{int(start_at * 1000)}-{int(finish_at * 1000)}",
                model_name=model_name,
                model_config=model_config,
                start_at=start_at,
                finish_at=finish_at,
                query=query,
                response=response,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                ext_info={**ext_info, **self.__uni_ext_info},
            )
        )


def _record_llm_query(
    model_name: str,
    model_config: Dict[str, Any],
    start_at: float,
    finish_at: float,
    query: Union[str, MessageList],
    response: Union[str, Message, List[Message]],
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    **ext_info,
):
    if _model_query_recorder is not None:
        # copy the query and response to avoid modification
        if isinstance(query, MessageList):
            query = MessageList(query.messages.copy())
        if isinstance(response, list):
            response = response.copy()

        _model_query_recorder.append(
            model_name,
            model_config,
            start_at,
            finish_at,
            query,
            response,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            **ext_info,
        )


class Model:
    @classmethod
    def make(cls, **kwargs):
        model_cls = cls
        if cls == Model:
            model_type = kwargs.pop("model_type")
            model_cls = eval(f"{model_type}Model")
        if "base_url" in kwargs and kwargs["base_url"].startswith("$"):
            if not (base_url := os.getenv(kwargs["base_url"][1:], None)):
                raise ValueError(f"{kwargs['base_url'][1:]} is not set")
            else:
                kwargs["base_url"] = base_url
        print_kwargs = {**kwargs, "api_key": "..."}
        rsu._ilog(f"Creating {model_cls}, config={print_kwargs}")
        return model_cls(**kwargs)

    @property
    def name(self) -> Optional[str]:
        return self._get_config().get("model_name")

    @rsu._abstractmethod
    def _get_config(self) -> dict:
        pass

    @rsu._abstractmethod
    def _ask_impl(
        self,
        messages: MessageList,
        use_official_config: bool,
        **kwargs,
    ) -> Message:
        pass

    def ask(
        self,
        messages: Union[MessageList, List[Message]],
        sampling_times=None,
        seed=None,
        use_official_config=False,
        **kwargs,
    ) -> Union[Message, List[Message]]:
        """
        Chat completion
        """
        kwargs["use_official_config"] = use_official_config
        if not isinstance(messages, MessageList):
            messages = MessageList(messages)
        if isinstance(seed, int):
            rsu._set_seed(seed)
        if not sampling_times:
            result = self._ask_impl(messages, **kwargs)
        elif isinstance(sampling_times, int):
            try:
                result = self._ask_impl(messages, n=sampling_times, **kwargs)
                if not isinstance(result, list) or len(result) != sampling_times:
                    raise NotImplementedError()
            except Exception as ex:
                result = [
                    self._ask_impl(messages, **kwargs) for _ in range(sampling_times)
                ]
        else:
            raise ValueError(
                f"`sampling_times` must be int or None, but got {sampling_times}"
            )
        return result


class OpenAIModel(Model):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        api_version: str = None,
        use_azure: bool = False,
        official_config: dict = None,
        **kwargs,
    ):
        if openai is None:
            raise ImportError("Install openai first: `pip install openai`")
        self.__base_url = base_url
        self.__api_key = api_key
        self.__api_version = api_version
        self.__use_azure = use_azure
        self.__model_name = model_name
        self.__ext_kwargs = kwargs
        self.__official_config = official_config or {}
        if not self.__use_azure:
            self.__openai_client = openai.Client(
                base_url=self.__base_url,
                api_key=self.__api_key,
            )
        else:
            self.__openai_client = openai.AzureOpenAI(
                azure_endpoint=self.__base_url,
                api_version=self.__api_version,
                api_key=self.__api_key,
            )

    def _get_config(self) -> dict:
        return {
            "model_type": "OpenAI",
            "base_url": "...",
            "api_key": "...",
            "api_version": "...",
            "use_azure": self.__use_azure,
            "model_name": self.__model_name,
            **self.__ext_kwargs,
        }

    def _ask_impl(
        self,
        messages: MessageList,
        use_official_config: bool,
        **kwargs,
    ) -> Union[Message, List[Message]]:
        def _not_context_length_exceeded(ex):
            if (
                isinstance(ex, openai.BadRequestError)
                and ex.code == "context_length_exceeded"
            ):
                rsu._wlog(f"Context is too long, stop retrying...")
                return False
            elif (  # For DeepSeek-V3
                isinstance(ex, openai.BadRequestError)
                and ex.code == "invalid_request_error"
                and "This model's maximum context length is 65536 tokens. However, you requested"
                in ex.message
            ):
                rsu._wlog(f"Context is too long, stop retrying...")
                return False
            elif (  # For vLLM
                isinstance(ex, openai.BadRequestError)
                and "This model's maximum context length is" in ex.message
                and "tokens. However, you requested" in ex.message
            ):
                rsu._wlog(f"Context is too long, stop retrying...")
                return False

            return True

        @retry(
            wait=wait_random_exponential(min=1, max=60 * 2),
            stop=stop_after_attempt(10),
            retry=retry_if_exception(_not_context_length_exceeded),
            reraise=True,
        )
        def _try_query(messages: MessageList):
            try:
                start_at = time.time()
                openai_response = self.__openai_client.chat.completions.create(
                    model=self.__model_name,
                    messages=messages.aslist(),
                    **{
                        **(
                            self.__ext_kwargs
                            if not use_official_config
                            else self.__official_config
                        ),
                        **kwargs,
                    },
                )
                if "n" in kwargs:
                    response = [
                        Message.from_dict(
                            {
                                **choice.message.to_dict(),
                                "logprobs": (
                                    choice.logprobs.to_dict()
                                    if choice.logprobs is not None
                                    else None
                                ),
                            }
                        )
                        for choice in openai_response.choices
                    ]
                else:
                    response = Message.from_dict(
                        {
                            **openai_response.choices[0].message.to_dict(),
                            "logprobs": (
                                openai_response.choices[0].logprobs.to_dict()
                                if openai_response.choices[0].logprobs is not None
                                else None
                            ),
                        }
                    )
                finish_at = time.time()
                return response, start_at, finish_at, openai_response
            except Exception as ex:
                rsu._wlog(f"OpenAI API error: {ex}, retrying...")
                raise ex

        if not isinstance(messages, MessageList):
            raise ValueError(f"`messages` must be a `MessageList`, but got {type(messages)}")

        try:
            response, start_at, finish_at, openai_response = _try_query(messages)
        except Exception as ex:
            if not _not_context_length_exceeded(ex):
                raise ContextLengthExceededException() from ex
            else:
                raise ex

        _record_llm_query(
            model_name=self.__model_name,
            model_config=self._get_config(),
            start_at=start_at,
            finish_at=finish_at,
            query=messages,
            response=response,
            prompt_tokens=openai_response.usage.prompt_tokens,
            completion_tokens=openai_response.usage.completion_tokens,
            total_tokens=openai_response.usage.total_tokens,
            _kwargs=kwargs,
            _original_openai_response=openai_response.to_dict(),
        )

        return response


@lru_cache(maxsize=None)
def make_model(config_file: str) -> Model:
    assert config_file.endswith(".json")
    return Model.make(**rsu._load_json(config_file))
