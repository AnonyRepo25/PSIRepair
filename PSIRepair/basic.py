from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, asdict, field


@dataclass
class CodeLocation:
    """
    Code location in a project
    """

    file_path: str = None
    lineno: int = None
    method_path: str = None


@dataclass
class BugInfo:
    project: Optional[Any] = None  # "Project"
    bug_locations: List[CodeLocation] = None
    num_buggy_lines: int = None
    is_single_line_bug: bool = None
    is_single_method_bug: bool = None
    is_single_file_bug: bool = None
    all_tests: Optional[List[str]] = None
    relevant_tests: Optional[List[str]] = None
    trigger_tests: List[str] = None
    _interal_id: str = None  # Specific to the project (e.g. {pid}_{vid} for Defects4J and RWB)

    def get_short_name(self) -> str:
        return f"{self.project.__class__.__name__}({self._interal_id})"

    def get_itendifier(self) -> str:
        return f"{self._interal_id}"


@dataclass
class Message:
    role: str
    content: str = None
    logprobs: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(
            role=d["role"],
            content=d.get("content", None),
            logprobs=d.get("logprobs", None),
        )

    @classmethod
    def _make_content(cls, role: str, content: str, **kwargs):
        content = content.format_map(kwargs) if kwargs else content
        return cls(role=role, content=content)

    @classmethod
    def system(cls, content: str, **kwargs):
        return cls._make_content(role="system", content=content, **kwargs)

    @classmethod
    def user(cls, content: str, **kwargs):
        return cls._make_content(role="user", content=content, **kwargs)

    @classmethod
    def assistant(cls, content: str, **kwargs):
        return cls._make_content(role="assistant", content=content, **kwargs)


@dataclass
class MessageList:
    messages: List[Message] = field(default_factory=list)

    def __getitem__(self, idx):
        return self.messages[idx]

    def append(self, msg: Message):
        self.messages.append(msg)

    def aslist(self) -> List[Dict[str, str]]:
        return [asdict(m) for m in self.messages]


@dataclass
class LLMQueryRecord:
    id: str
    model_name: str
    model_config: Dict[str, Any]
    start_at: float
    finish_at: float
    query: Union[str, MessageList]
    response: Union[str, Message, List[Message]]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    ext_info: Dict[str, Any] = None
