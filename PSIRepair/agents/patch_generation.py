from os import getenv
from io import StringIO
from dataclasses import dataclass, asdict
from typing import Any, Optional, Dict, List, Iterator
from .. import rs_utils as rsu
from ..basic import BugInfo, Message, LLMQueryRecord
from ..utils import (
    _add_code_block,
    _parse_code_block,
    _make_apr_task_input,
    _make_func_body_with_bug_location,
    _assert_bug_is_single_method,
)
from ..model import Model, ContextLengthExceededException
from .agent import Agent, AgentAskRecord, USE_JAVA_CODEBLOCK_INSTRUCTION
from .code_snippet_retrieval import CodeSnippet, GraphBasedCodeSnippetRetrievalAgent
from .error_resolution import RetrievalAugmentedTestErrorResolutionAgent


@dataclass
class GeneratedPatch:
    patch_file_path: str
    patch_md_path: str
    patch_md_code: str


class PatchGenerationAgent(Agent):
    @dataclass
    class Input:
        bug: BugInfo

    @dataclass
    class Output:
        found_plausible: bool
        generated_patches: List[str]
        valid_results: List[dict]
        patch_file_path: str
        patch_md_path: str
        plausible_patch_md_code: Optional[str]
        llm_query_records: List[LLMQueryRecord] = None
        asked_agent_outputs: List[AgentAskRecord] = None

    def __init__(self, name: Optional[str] = None):
        super().__init__(name or self.__class__.__name__)

    @rsu._abstractmethod
    def _generate_patches(
        input: "Input",
        valid_results: List[dict],
    ) -> Iterator[GeneratedPatch]:
        pass

    def _ask_impl(self, input: "Input") -> "Output":

        bug = input.bug
        patches = []
        found_plausible = False
        plausible_patch = None
        validated_pacthes = set()
        valid_results = []

        for i, patch_info in enumerate(self._generate_patches(input, valid_results)):
            patches.append(patch_info)

            # Validate the patch
            self.ilog(f"Val@{i} -- Validating a patch")

            patch = patch_info.patch_md_code
            if patch in validated_pacthes:
                self.wlog(f"Found a repeated patch, SKIP")
                valid_results.append(
                    next(
                        v.copy()
                        for v in valid_results
                        if v["patch"]["patch_md_code"] == patch
                    )
                )
                continue

            validated_pacthes.add(patch)
            compilation_result = None
            validation_result = None
            with bug.project.apply_patch(
                file_path=patch_info.patch_file_path,
                method_path=patch_info.patch_md_path,
                patch=patch_info.patch_md_code,
            ):
                ret, stdout, stderr = bug.project.run_compile()
                compilation_result = {
                    "success": ret == 0,
                    "stdout": stdout,
                    "stderr": stderr,
                }
                if ret == 0:
                    validation_result = bug.project.run_test(
                        relevant_bug=bug, strategy="trigger->all"
                    )
                valid_results.append(
                    {
                        "patch": asdict(patch_info),
                        "compilation_result": compilation_result,
                        "validation_result": (
                            asdict(validation_result) if validation_result else None
                        ),
                    }
                )
                if validation_result and validation_result.passed:
                    self.ilog(f"Found one plausible patch for {bug.get_short_name()}")
                    found_plausible = True
                    plausible_patch = patch_info
                    break

        assert not found_plausible or plausible_patch is not None

        return self.Output(
            found_plausible=found_plausible,
            generated_patches=[p.patch_md_code for p in patches],
            valid_results=valid_results,
            patch_file_path=bug.bug_locations[0].file_path,
            patch_md_path=bug.bug_locations[0].method_path,
            plausible_patch_md_code=(
                plausible_patch.patch_md_code if plausible_patch else None
            ),
        )


class RetrievalAugmentedPatchGenerationAgent(PatchGenerationAgent):
    @dataclass
    class Input(PatchGenerationAgent.Input):
        pass

    @dataclass
    class Output(PatchGenerationAgent.Output):
        pass

    def __init__(
        self,
        model: Model,
        max_num_patches_per_bug: int,
        # For GraphBasedCodeSnippetRetrievalAgent
        r_local_cache_dir: str,
        r_embedding_model_config: Dict[str, Any],
        r_num_retrieved_subgraphs: Optional[int] = None,
        r_num_retrieved_nodes_per_subgraph: Optional[int] = None,
        r_num_group_of_retrieved: Optional[int] = None,
        r_num_retrieved_nodes_per_group: Optional[int] = None,
        r_ratio_of_retrieved_fields_nodes: Optional[str] = None,
        r_subgraph_selection_strategy: Optional[str] = None,
        r_rerank_model: Optional[Model] = None,
        r_num_rerank_tries_per_retrieved_node: Optional[int] = None,
        # For RetrievalAugmentedTestErrorResolutionAgent
        max_num_error_resolution_attempts: Optional[int] = None,
    ):
        super().__init__(name="RAPGA")
        self.__model = model
        self.__max_num_patches_per_bug = max_num_patches_per_bug

        self.__code_snippet_retrieval_agent = GraphBasedCodeSnippetRetrievalAgent(
            embedding_model_config=r_embedding_model_config,
            num_retrieved_subgraphs=r_num_retrieved_subgraphs,
            num_retrieved_nodes_per_subgraph=r_num_retrieved_nodes_per_subgraph,
            num_group_of_retrieved=r_num_group_of_retrieved,
            num_retrieved_nodes_per_group=r_num_retrieved_nodes_per_group,
            ratio_of_retrieved_fields_nodes=r_ratio_of_retrieved_fields_nodes,
            subgraph_selection_strategy=r_subgraph_selection_strategy,
            rerank_model=r_rerank_model or model,
            num_rerank_tries_per_retrieved_node=r_num_rerank_tries_per_retrieved_node,
            num_contextual_retrieved_nodes=r_num_retrieved_nodes_per_subgraph,
            local_cache_dir=r_local_cache_dir,
        )

        self.__error_resolution_agent = RetrievalAugmentedTestErrorResolutionAgent(
            model=model,
            r_embedding_model_config=r_embedding_model_config,
            r_local_cache_dir=r_local_cache_dir,
            r_num_retrieved_subgraphs=r_num_retrieved_subgraphs,
            r_num_retrieved_nodes_per_subgraph=int(
                getenv("ERA_NRNpSG") or r_num_retrieved_nodes_per_subgraph
            ),
            r_num_group_of_retrieved=1,  # NOTE: Only one group is enough for RATErrResAgent
            r_num_retrieved_nodes_per_group=r_num_retrieved_nodes_per_group,
            r_ratio_of_retrieved_fields_nodes=r_ratio_of_retrieved_fields_nodes,
            r_subgraph_selection_strategy=getenv("ERA_SGSS")
            or r_subgraph_selection_strategy,
            r_rerank_model=r_rerank_model or model,
            r_num_rerank_tries_per_retrieved_node=r_num_rerank_tries_per_retrieved_node,
        )
        self.__max_num_error_resolution_attempts = max_num_error_resolution_attempts

    def _make_system_prompt(self) -> str:
        return "You are an Automated Program Repair tool"

    def _make_prompt(
        self,
        bug: BugInfo,
        code_snippets: List[CodeSnippet],
        **kwargs,
    ) -> str:
        bug = _assert_bug_is_single_method(bug)

        task_prompt = _make_apr_task_input(
            bug=bug,
            with_line_numbers=False,
            simple_style=True,
            **kwargs,
        )

        if not code_snippets:
            return f"""\
Given the following buggy method, the triggered failed test method, and the failing message, please generate the correct full method to fix the bug.
{USE_JAVA_CODEBLOCK_INSTRUCTION}

{task_prompt}"""

        fmt = f"""\
Given the following buggy method, the triggered failed test method, and the failing message, please generate the correct full method to fix the bug.
{USE_JAVA_CODEBLOCK_INSTRUCTION.replace('{', '{{').replace('}', '}}')}

{{task_prompt}}

Additionally, you can refer to the following code snippets which are bug-free and extracted from the target project to guide the solution:
{{code_snippets}}"""

        # Make prompt: basic prompt + code snippets
        cs_prompt_sb = StringIO()
        for i, s in enumerate(code_snippets, start=1):
            path, content = s.path, s.content
            cs_prompt_sb.write(f"Snippet {i}. {path}\n")
            if s.additional_info.get("from", None) == "contextual":
                cs_prompt_sb.write(
                    f" - Relations with Bug method: {s.additional_info['forward_relationships']}\n"
                )
            cs_prompt_sb.write(_add_code_block(content, lang=bug.project.get_lang()))
            cs_prompt_sb.write("\n\n")
        return fmt.format(
            code_snippets=cs_prompt_sb.getvalue().strip(),
            task_prompt=task_prompt,
        )

    def _make_rerank_prompt_template(self, bug: BugInfo) -> str:
        bug = _assert_bug_is_single_method(bug)
        apr_task_prompt = _make_apr_task_input(
            bug=bug,
            with_line_numbers=False,
            simple_style=True,
        )

        template = f"""\
Given the following buggy method, the triggered failed test method, the failing message, and a code snippet which are bug-free and extracted from the target project to guide the solution.
Before generate the correct method to fix the bug, consider whether the provided code snnipet is helpful for repair the bug.
<<<<|output_format|>>>>

{apr_task_prompt}

Code snippet:
```{bug.project.get_lang()}
<<<<|code_snippet|>>>>
```"""

        return (
            template.replace("{", "{{")
            .replace("}", "}}")
            .replace("<<<<|", "{")
            .replace("|>>>>", "}")
        )

    def _generate_patches(
        self,
        input: "Input",
        valid_results: List[dict],
    ) -> Iterator[GeneratedPatch]:
        # TODO: do not directly use CodeToolkitBackend here
        from ..tools.code_toolkit import CodeToolkitBackend

        bug = _assert_bug_is_single_method(input.bug)

        bug_method = _make_func_body_with_bug_location(
            bug=bug,
            with_bug_location=True,
            with_line_numbers=False,
            additional_comment_lines=None,
        )

        bug_file_path = (
            f"{bug.project.get_src_sourcepath()}/{bug.bug_locations[0].file_path}"
        )
        bug_md_path = bug.bug_locations[0].method_path
        bug_md_start_l, bug_md_end_l = map(
            int, bug_md_path.split("[")[-1].split("]")[0].split("-")
        )
        _path_type, qualified_name = CodeToolkitBackend.getPkgClsMdPathByPosition(
            bug_file_path,  # file path
            bug_md_start_l,  # start line
            bug_md_end_l,  # end line
            True,  # return qualified name
        )
        if _path_type != "method":
            # NOTE: Bug in spoon: pos.getLine() is not the actual start line of the method in some cases
            self.wlog("Failed to get qualname by [start,end], fallback to [end,end]")
            self.wlog(f">>> file: {bug_file_path}")
            self.wlog(f">>> [start, end]: [{bug_md_start_l}, {bug_md_end_l}]")
            _path_type, qualified_name = CodeToolkitBackend.getPkgClsMdPathByPosition(
                bug_file_path,  # file path
                bug_md_end_l,  # start line
                bug_md_end_l,  # end line
                True,  # return qualified name
            )
        qualified_name = str(qualified_name)  # java.lang.String -> str
        assert _path_type == "method"
        del _path_type

        query = bug_method
        code_snippets = self.__code_snippet_retrieval_agent.ask(
            self.__code_snippet_retrieval_agent.Input(
                project=bug.project,
                focal_method_qualified_name=qualified_name,
                query=query,
                rerank_prompt=self._make_rerank_prompt_template(bug),
            )
        ).code_snippets.copy()
        self.ilog(f"Retrieved {sum(len(c) for _, c in code_snippets)} code snippets")
        self.ilog(f">> num levels: {len(code_snippets)}")
        for i, (lv, c) in enumerate(code_snippets):
            self.ilog(f">>> level {i}-{lv}: {len(c)} code snippets")

        code_snippets.append(("fallback", []))
        self.ilog(f"Added `fallback`: <dummy code snippet>")

        max_num_patches = self.__max_num_patches_per_bug
        max_num_patches_per_level = max_num_patches // len(code_snippets)
        assert max_num_patches == max_num_patches_per_level * len(code_snippets)
        for i, (i_level, i_code_snippets) in enumerate(code_snippets):
            self.ilog(f"Try@{i} -- Generating patches for level `{i_level}`")
            num_i_code_snippets = len(i_code_snippets)
            system_prompt = self._make_system_prompt()
            while True:
                try:
                    prompt = self._make_prompt(
                        bug=bug,
                        code_snippets=i_code_snippets[:num_i_code_snippets],
                    )
                    assert self.__max_num_error_resolution_attempts is not None
                    assert (
                        max_num_patches_per_level
                        % (self.__max_num_error_resolution_attempts + 1)
                        == 0
                    )
                    num_attempts = max_num_patches_per_level // (
                        self.__max_num_error_resolution_attempts + 1
                    )
                    num_er_attempts = self.__max_num_error_resolution_attempts
                    for j in range(num_attempts):
                        self.ilog(f"Try@{i}@{j} -- Generating a patch")
                        messages = [
                            Message.system(system_prompt),
                            Message.user(prompt),
                        ]
                        response = self.__model.ask(messages)
                        yield GeneratedPatch(
                            patch_file_path=bug.bug_locations[0].file_path,
                            patch_md_path=bug.bug_locations[0].method_path,
                            patch_md_code=_parse_code_block(
                                response.content,
                                lang=bug.project.get_lang(),
                            ),
                        )
                        for k in range(num_er_attempts):
                            self.ilog(f"Try@{i}@{j}@{k} -- Regenerating a patch")

                            val_r = valid_results[-1].copy()
                            assert val_r is not None

                            prev_patch = val_r["patch"]["patch_md_code"]
                            compile_r = val_r["compilation_result"]
                            test_r = val_r["validation_result"]

                            if not val_r["compilation_result"]["success"]:
                                failed_reason = "compile_error"
                                failure_log = compile_r["stderr"]
                                test_md = None
                            elif not val_r["validation_result"]["passed"]:
                                failed_reason = "test_failed"
                                if not val_r["validation_result"]["timeout"]:
                                    failed_tests = test_r["failing_tests"]
                                    assert len(failed_tests) >= 1
                                    test_md_path = failed_tests[0]["md_path"]
                                    test_md = "\n".join(
                                        [
                                            line
                                            for _, line in bug.project.get_test_method_code_lines(
                                                test_method_full_path=test_md_path,
                                                return_file_path=False,
                                            )
                                            or [(-1, "missing")]
                                        ]
                                    )
                                    failure_log = failed_tests[0]["traceback"]
                                else:
                                    failed_reason = "test_failed"
                                    test_md = "// Error: Timeout when running tests"
                                    failure_log = "Timeout when running tests"

                            er_input = self.__error_resolution_agent.Input(
                                project=bug.project,
                                bug_method_qualified_name=qualified_name,
                                bug_method=bug_method,
                                failed_reason=failed_reason,
                                failed_fix_attempt=prev_patch,
                                failed_test_method=test_md,
                                test_failure_log=failure_log,
                            )
                            r = self.__error_resolution_agent.ask(er_input)
                            yield GeneratedPatch(
                                patch_file_path=bug.bug_locations[0].file_path,
                                patch_md_path=bug.bug_locations[0].method_path,
                                patch_md_code=r.new_fix_attempt,
                            )
                    break  # break while True
                except ContextLengthExceededException as ex:
                    if num_i_code_snippets == 0:
                        raise ex  # let it crash
                    self.wlog("The prompt is too long")
                    self.wlog(f">>>> Try to reduce the number of code snippets")
                    self.wlog(f">>>>> From {num_i_code_snippets}")
                    num_i_code_snippets -= 1
                    self.wlog(f">>>>> To {num_i_code_snippets}")
