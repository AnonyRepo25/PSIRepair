from typing import List, Dict, Any
from dataclasses import dataclass
from .. import rs_utils as rsu
from ..model import LLMQueryRecorder
from ..utils import _is_debug_mode

USE_JAVA_CODEBLOCK_INSTRUCTION = "Use a Java code block to write your response. For example:\n```java\npublic static int sum(int a, int b) {\n    return a + b;\n}\n```"

_agent_ask_recorder = None


@dataclass
class AgentAskRecord:
    agent_name: str
    agent_itendifier: str
    agent_output: Any


class AgentAskRecorder:
    def __init__(self):
        self.__records: List[AgentAskRecord] = []

    @property
    def records(self):
        return self.__records

    def __enter__(self):
        global _agent_ask_recorder
        self.__old_recorder = _agent_ask_recorder
        _agent_ask_recorder = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global _agent_ask_recorder
        _agent_ask_recorder = self.__old_recorder
        return False

    def append(self, agent_name: str, agent_identifier: str, agent_output: Any):
        self.__records.append(
            AgentAskRecord(
                agent_name=agent_name,
                agent_itendifier=agent_identifier,
                agent_output=agent_output,
            )
        )


def _record_agent_ask(agent_name: str, agent_identifier: str, agent_output: Any):
    global _agent_ask_recorder
    if _agent_ask_recorder is not None:
        _agent_ask_recorder.append(agent_name, agent_identifier, agent_output)


class Agent:
    _agent_id_pool: Dict[str, int] = {}

    @classmethod
    def _make_agent_itendifier(cls, agent_name: str):
        if agent_name not in cls._agent_id_pool:
            cls._agent_id_pool[agent_name] = 0
        agent_id = cls._agent_id_pool[agent_name]
        agent_itendifier = f"{agent_name}@{agent_id}"
        cls._agent_id_pool[agent_name] += 1
        return agent_itendifier

    def __init__(self, name) -> None:
        self.__name = name
        self.__itendifier = self._make_agent_itendifier(name)

    @rsu._abstractmethod
    def _ask_impl(self, input):
        pass

    def ask(self, input, sampling_times=None, seed=None):
        if _is_debug_mode():
            rsu._ilog(f"Asking agent(`{self.__itendifier}`) >>>> start <<<<")

        if isinstance(seed, int):
            rsu._set_seed(seed)
        if not sampling_times:
            with LLMQueryRecorder() as llm_recorder, AgentAskRecorder() as agent_recorder:
                output = self._ask_impl(input)
            if not hasattr(output, "llm_query_records"):
                raise ValueError("`llm_query_records` is required but None")
            if not hasattr(output, "asked_agent_outputs"):
                raise ValueError("`asked_agent_outputs` is required but None")
            output.llm_query_records = llm_recorder.records
            output.asked_agent_outputs = agent_recorder.records
            _record_agent_ask(self.__name, self.__itendifier, output)
        elif isinstance(sampling_times, int):
            output = []
            for _ in range(sampling_times):
                with LLMQueryRecorder() as llm_recorder:
                    one_output = self._ask_impl(input)
                if not hasattr(one_output, "llm_query_records"):
                    raise ValueError("`llm_query_records` is None")
                if not hasattr(one_output, "asked_agent_outputs"):
                    raise ValueError("`asked_agent_outputs` is None")
                one_output.llm_query_records = llm_recorder.records
                one_output.asked_agent_outputs = agent_recorder.records
                _record_agent_ask(self.__name, self.__itendifier, one_output)
                output.append(one_output)
        else:
            raise ValueError(
                f"`sampling_times` must be int or None, but got {sampling_times}"
            )

        if _is_debug_mode():
            rsu._ilog(f"Asking agent(`{self.__itendifier}`) >>>> finished <<<<")

        return output

    def _log(self, log_fn, *msg, fback, **kwargs):
        log_fn(f"Agent[{self.__itendifier}]", *msg, fback=fback, **kwargs)

    def ilog(self, *args, **kwargs):
        self._log(rsu._ilog, *args, fback=3, **kwargs)

    def wlog(self, *args, **kwargs):
        self._log(rsu._wlog, *args, fback=3, **kwargs)

    def elog(self, *args, **kwargs):
        self._log(rsu._elog, *args, fback=3, **kwargs)

    def flog(self, *args, **kwargs):
        self._log(rsu._flog, *args, fback=3, **kwargs)
