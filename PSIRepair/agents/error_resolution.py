from io import StringIO
from dataclasses import dataclass
from typing import Any, Optional, Dict, List
from ..basic import Message, LLMQueryRecord
from ..utils import _add_code_block, _parse_code_block
from ..model import Model, ContextLengthExceededException
from .agent import Agent, AgentAskRecord, USE_JAVA_CODEBLOCK_INSTRUCTION
from .code_snippet_retrieval import CodeSnippet, GraphBasedCodeSnippetRetrievalAgent


class RetrievalAugmentedTestErrorResolutionAgent(Agent):
    @dataclass
    class Input:
        project: Any  # only for CodeRetrievalAgent
        bug_method_qualified_name: str  # only for CodeRetrievalAgent
        bug_method: str
        failed_reason: str  # ["compile_error", "test_failed"]
        failed_fix_attempt: str
        failed_test_method: Optional[str]
        test_failure_log: str

    @dataclass
    class Output:
        project_name: str
        bug_method_qualified_name: str
        bug_method: str
        failed_reason: str
        failed_fix_attempt: str
        failed_test_method: Optional[str]
        test_failure_log: str

        new_fix_attempt: str
        llm_query_records: List[LLMQueryRecord] = None
        asked_agent_outputs: List[AgentAskRecord] = None

    def __init__(
        self,
        model: Model,
        # For GraphBasedCodeSnippetRetrievalAgent
        r_embedding_model_config: Dict[str, Any],
        r_local_cache_dir: str,
        r_num_retrieved_subgraphs: int,
        r_num_retrieved_nodes_per_subgraph: int,
        r_num_group_of_retrieved: int,
        r_num_retrieved_nodes_per_group: int,
        r_ratio_of_retrieved_fields_nodes: float,
        r_subgraph_selection_strategy: str,
        r_rerank_model: Model,
        r_num_rerank_tries_per_retrieved_node: int,
    ):
        super().__init__(name="RATestErrResA")
        self.__model = model
        self.__code_snippet_retrieval_agent = GraphBasedCodeSnippetRetrievalAgent(
            embedding_model_config=r_embedding_model_config,
            num_retrieved_subgraphs=r_num_retrieved_subgraphs,
            num_retrieved_nodes_per_subgraph=r_num_retrieved_nodes_per_subgraph,
            num_group_of_retrieved=r_num_group_of_retrieved,
            num_retrieved_nodes_per_group=r_num_retrieved_nodes_per_group,
            ratio_of_retrieved_fields_nodes=r_ratio_of_retrieved_fields_nodes,
            subgraph_selection_strategy=r_subgraph_selection_strategy,
            rerank_model=r_rerank_model,
            num_rerank_tries_per_retrieved_node=r_num_rerank_tries_per_retrieved_node,
            num_contextual_retrieved_nodes=None,
            local_cache_dir=r_local_cache_dir,
        )

    def _make_task_input_prompt(self, input: "Input") -> str:
        bug_md_qualified_name = input.bug_method_qualified_name
        bug_method = input.bug_method
        failed_reason = input.failed_reason
        failed_fix_attempt = input.failed_fix_attempt
        failed_test_method = input.failed_test_method
        test_failure_log = input.test_failure_log

        if failed_reason == "compile_error":
            assert failed_test_method is None
            task_input_prompt = f"""\
Buggy Method (the bug lines are marked with '// bug line'):
```java
// {bug_md_qualified_name}
{bug_method}
```

Failed Fix Attempt:
```java
{failed_fix_attempt}
```

Compilation Error Log:
```
{test_failure_log}
```"""
        elif failed_reason == "test_failed":
            assert failed_test_method is not None
            task_input_prompt = f"""\
Buggy Method (the bug lines are marked with '// bug line'):
```java
// {bug_md_qualified_name}
{bug_method}
```

Failed Fix Attempt:
```java
{failed_fix_attempt}
```

Failed Test Method:
```java
{failed_test_method}
```

Test Failure Log:
```
{test_failure_log}
```"""
        else:
            raise ValueError(f"Unknown failed_reason: {failed_reason}")

        return task_input_prompt

    def _make_system_prompt(self) -> str:
        return "You are an Automated Program Repair tool"

    def _make_prompt(self, input: "Input", code_snippets: List[CodeSnippet]) -> str:
        failed_reason = input.failed_reason
        failed_test_method = input.failed_test_method
        task_input_prompt = self._make_task_input_prompt(input=input)

        if code_snippets:
            cs_prompt_sb = StringIO()
            cs_prompt_sb.write("\n")
            cs_prompt_sb.write(
                "You can refer to the following code snippets which are "
                + "bug-free and extracted from the target project to guide the solution:\n"
            )
            for i, s in enumerate(code_snippets, start=1):
                path, content = s.path, s.content
                cs_prompt_sb.write(f"Snippet {i}. {path}\n")
                cs_prompt_sb.write(_add_code_block(content, lang="java"))
                cs_prompt_sb.write("\n\n")
            code_snippets_prompt = cs_prompt_sb.getvalue()
            del code_snippets, cs_prompt_sb, i, s, path, content
        else:
            code_snippets_prompt = ""

        if failed_reason == "compile_error":
            assert failed_test_method is None
            prompt = f"""\
Given the following buggy method, the failed fix attempt, and the compilation error log, please generate the correct full method to pass the compilation and fix the bug.
{USE_JAVA_CODEBLOCK_INSTRUCTION.replace('{', '{{').replace('}', '}}')}

{task_input_prompt}
{code_snippets_prompt}
Now! Generate the correct full method. Let's think step by step."""
        elif failed_reason == "test_failed":
            assert failed_test_method is not None
            prompt = f"""\
Given the following buggy method, the failed fix attempt, the failed test method, and the test failure log, please generate the correct full method to pass the test and fix the bug.
{USE_JAVA_CODEBLOCK_INSTRUCTION.replace('{', '{{').replace('}', '}}')}

{task_input_prompt}
{code_snippets_prompt}
Now! Generate the correct full method. Let's think step by step."""
        else:
            raise ValueError(f"Unknown failed_reason: {failed_reason}")
        return prompt

    def _make_rerank_prompt_template(self, input: "Input") -> str:
        failed_reason = input.failed_reason
        task_input_prompt = self._make_task_input_prompt(input=input)

        if failed_reason == "compile_error":
            rerank_prompt_template = f"""\
Given the following buggy method, the failed fix attempt, and the compilation error log, and a code snippet which are bug-free and extracted from the target project to guide the solution.
Before generate the correct method to pass the compilation and fix the bug, consider whether the provided code snnipet is helpful for resolve the compilation error and fix the bug.
<<<<|output_format|>>>>

{task_input_prompt}

Code snippet:
```java
<<<<|code_snippet|>>>>
```"""
        elif failed_reason == "test_failed":
            rerank_prompt_template = f"""\
Given the following buggy method, the failed fix attempt, the failed test method, and the test failure log, and a code snippet which are bug-free and extracted from the target project to guide the solution.
Before generate the correct method to pass the test and fix the bug, consider whether the provided code snnipet is helpful for resolve the test failure and fix the bug.
<<<<|output_format|>>>>

{task_input_prompt}

Code snippet:
```java
<<<<|code_snippet|>>>>
```"""
        else:
            raise ValueError(f"Unknown failed_reason: {failed_reason}")

        return (
            rerank_prompt_template.replace("{", "{{")
            .replace("}", "}}")
            .replace("<<<<|", "{")
            .replace("|>>>>", "}")
        )

    def _ask_impl(self, input: "Input") -> "Output":
        self.ilog("Generating a new fix attempt")
        self.ilog(f">>>> Bug Method: {input.bug_method_qualified_name}")
        self.ilog(f">>>> Failed Reason: {input.failed_reason}")

        query_prompt = self._make_prompt(input=input, code_snippets=[])

        ra_output = self.__code_snippet_retrieval_agent.ask(
            self.__code_snippet_retrieval_agent.Input(
                project=input.project,
                focal_method_qualified_name=input.bug_method_qualified_name,
                query=query_prompt,
                rerank_prompt=self._make_rerank_prompt_template(input),
            )
        )
        code_snippets = ra_output.code_snippets
        assert len(code_snippets) == 1, "Only one group is expected"
        code_snippets = [cs for cs in code_snippets[0][1] if cs.content != "N/A"]

        del query_prompt, ra_output

        while True:
            try:
                system_prompt = self._make_system_prompt()
                prompt = self._make_prompt(
                    input=input,
                    code_snippets=code_snippets,
                )
                messages = [Message.system(system_prompt), Message.user(prompt)]
                response = self.__model.ask(messages)
                break
            except ContextLengthExceededException as ex:
                if len(code_snippets) == 0:
                    raise ex
                self.wlog("The prompt is too long")
                self.wlog(f">>>> Try to reduce the number of code snippets")
                self.wlog(f">>>>> From {len(code_snippets)}")
                code_snippets = code_snippets[:-1]
                self.wlog(f">>>>> To {len(code_snippets)}")

        return self.Output(
            project_name=input.project.get_itendifier(ignore_project_version=False),
            bug_method_qualified_name=input.bug_method_qualified_name,
            bug_method=input.bug_method,
            failed_reason=input.failed_reason,
            failed_fix_attempt=input.failed_fix_attempt,
            failed_test_method=input.failed_test_method,
            test_failure_log=input.test_failure_log,
            new_fix_attempt=_parse_code_block(response.content, lang="java"),
        )
