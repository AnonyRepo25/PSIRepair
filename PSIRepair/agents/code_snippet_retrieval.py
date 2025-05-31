import os
import tqdm
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Tuple, List
from ..rs_utils import _save_as_json, _split_list_as_chunks
from ..basic import LLMQueryRecord
from ..utils import (
    SkipException,
    _generate_sha256_key,
    _is_debug_mode,
)
from ..model import Message, Model
from ..project import Project
from ..vector_store import CodeSnippetVectorStore
from .agent import Agent, AgentAskRecord


@dataclass
class CodeSnippet:
    type: str
    path: str  # method path or class path
    content: str
    file_path: Optional[str]
    langchain_doc: Optional[Dict[str, Any]]
    additional_info: Optional[Dict[str, Any]] = field(default_factory=dict)


class GraphBasedCodeSnippetRetrievalAgent(Agent):
    _avaliable_subgraph_selection_strategies = [
        "only_max_subgraph",
        "contextual_and_similarity",
    ]

    @dataclass
    class Input:
        project: Project
        focal_method_qualified_name: str
        query: str
        rerank_prompt: Optional[str] = None

    @dataclass
    class Output:
        project_name: str
        focal_method_qualified_name: str
        query: str
        rerank_prompt: Optional[str]

        unranked_code_snippets: List[CodeSnippet]
        code_snippets: List[Tuple[str, List[CodeSnippet]]]  # group -> code_snippets
        llm_query_records: List[LLMQueryRecord] = None
        asked_agent_outputs: List[AgentAskRecord] = None

    @dataclass
    class Subgraph:
        name: str
        nodes: List[Any]

    def __init__(
        self,
        embedding_model_config: Dict[str, Any],
        num_retrieved_subgraphs: int,
        num_retrieved_nodes_per_subgraph: int,
        num_group_of_retrieved: int,
        num_retrieved_nodes_per_group: int,
        ratio_of_retrieved_fields_nodes: str,  # float or fraction or "auto"
        subgraph_selection_strategy: str,
        rerank_model: Optional[Model],
        num_rerank_tries_per_retrieved_node: Optional[int],
        num_contextual_retrieved_nodes: Optional[int],
        local_cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__(name="GraphBasedCSRA")

        if subgraph_selection_strategy == "contextual_and_similarity":
            if ratio_of_retrieved_fields_nodes != "auto":
                raise ValueError(
                    "When `subgraph_selection_strategy` is `contextual_and_similarity`, "
                    + "`ratio_of_retrieved_fields_nodes` must be `auto`."
                )
            if num_retrieved_subgraphs != 1:
                raise ValueError(
                    "When `subgraph_selection_strategy` is `contextual_and_similarity`, "
                    + "`num_retrieved_subgraphs` must be 1."
                )
            if num_retrieved_nodes_per_subgraph != num_contextual_retrieved_nodes:
                raise ValueError(
                    "When `subgraph_selection_strategy` is `contextual_and_similarity`, "
                    + "`num_retrieved_nodes_per_subgraph` must be equal to `num_contextual_retrieved_nodes`."
                )
            if (
                num_retrieved_nodes_per_subgraph
                != num_contextual_retrieved_nodes
                != num_group_of_retrieved * num_retrieved_nodes_per_group
            ):
                raise ValueError(
                    "When `subgraph_selection_strategy` is `contextual_and_similarity`, "
                    + "`num_retrieved_nodes_per_subgraph`, `num_contextual_retrieved_nodes`, "
                    + "and `num_group_of_retrieved` * `num_retrieved_nodes_per_group` "
                    + "must be equal to each other."
                )
            if rerank_model is None or num_rerank_tries_per_retrieved_node is None:
                raise ValueError(
                    "When `subgraph_selection_strategy` is `contextual_and_similarity`, "
                    + "`rerank_model` and `num_rerank_tries_per_retrieved_node` must be provided."
                )

        if (
            num_retrieved_subgraphs < 1
            or num_retrieved_nodes_per_subgraph < 1
            or num_group_of_retrieved < 1
            or num_retrieved_nodes_per_group < 1
        ):
            raise ValueError(
                "`num_retrieved_subgraphs`, `num_retrieved_nodes_per_subgraph`, "
                + "`num_group_of_retrieved`, and `num_retrieved_nodes_per_group` "
                + "must be greater than 0."
            )

        if (
            num_retrieved_nodes_per_group * num_group_of_retrieved
            > num_retrieved_nodes_per_subgraph * num_retrieved_subgraphs
        ):
            raise ValueError(
                "`num_retrieved_nodes_per_group` * `num_group_of_retrieved` "
                + "must be less than or equal to "
                + "`num_retrieved_nodes_per_subgraph` * `num_retrieved_subgraphs`."
                + f"Got {num_retrieved_nodes_per_group} * {num_group_of_retrieved} = {num_retrieved_nodes_per_group * num_group_of_retrieved} "
                + f"and {num_retrieved_nodes_per_subgraph} * {num_retrieved_subgraphs} = {num_retrieved_nodes_per_subgraph * num_retrieved_subgraphs}."
            )

        if (ratio_of_retrieved_fields_nodes != "auto") and (
            eval(ratio_of_retrieved_fields_nodes) < 0
            or eval(ratio_of_retrieved_fields_nodes) > 1
        ):
            raise ValueError(
                "`ratio_of_retrieved_fields_nodes` must be in the range [0, 1] or `auto`. "
                + f"Got ratio_of_retrieved_fields_nodes={ratio_of_retrieved_fields_nodes}"
            )

        if (
            subgraph_selection_strategy
            not in self._avaliable_subgraph_selection_strategies
        ):
            raise ValueError(
                f"Unknown subgraph_selection_strategy: {subgraph_selection_strategy}"
                + f", available strategies: {self._avaliable_subgraph_selection_strategies}"
            )

        if not isinstance(rerank_model, Model):
            raise ValueError(
                "`rerank_model` must be a `Model` instance."
                + f"Got {rerank_model} of type {type(rerank_model)}."
            )
        if not isinstance(num_rerank_tries_per_retrieved_node, int):
            raise ValueError(
                "`num_rerank_tries_per_retrieved_node` must be an integer."
                + f"Got {num_rerank_tries_per_retrieved_node} of type {type(num_rerank_tries_per_retrieved_node)}."
            )
        elif num_rerank_tries_per_retrieved_node < 1:
            raise ValueError(
                "`num_rerank_tries_per_retrieved_node` must be greater than 0."
                + f"Got {num_rerank_tries_per_retrieved_node}."
            )

        self.__embedding_model_config = embedding_model_config.copy()
        self.__num_retrieved_subgraphs = num_retrieved_subgraphs
        self.__num_retrieved_nodes_per_subgraph = num_retrieved_nodes_per_subgraph
        self.__num_group_of_retrieved = num_group_of_retrieved
        self.__num_retrieved_nodes_per_group = num_retrieved_nodes_per_group
        self.__num_retrieved_fields_nodes_per_subgraph = (
            int(
                eval(
                    f"{num_retrieved_nodes_per_subgraph} * ({ratio_of_retrieved_fields_nodes})"
                )
            )
            if ratio_of_retrieved_fields_nodes != "auto"
            else None
        )
        self.__num_retrieved_method_nodes_per_subgraph = (
            (
                num_retrieved_nodes_per_subgraph
                - self.__num_retrieved_fields_nodes_per_subgraph
            )
            if ratio_of_retrieved_fields_nodes != "auto"
            else None
        )
        assert ratio_of_retrieved_fields_nodes == "auto" or (
            self.__num_retrieved_fields_nodes_per_subgraph
            + self.__num_retrieved_method_nodes_per_subgraph
            == self.__num_retrieved_nodes_per_subgraph
        )
        self.__subgraph_selection_strategy = subgraph_selection_strategy
        self.__rerank_model = rerank_model
        self.__num_rerank_tries_per_retrieved_node = num_rerank_tries_per_retrieved_node
        self.__num_contextual_retrieved_nodes = num_contextual_retrieved_nodes
        self.__ratio_of_retrieved_fields_nodes = ratio_of_retrieved_fields_nodes
        self.__local_cache_dir = (
            f"{local_cache_dir}/{_generate_sha256_key(f'graph-based--{self.__embedding_model_config}')}"
            if local_cache_dir is not None
            else None
        )

        if self.__local_cache_dir is not None:
            self.ilog(f"Using local cache dir: {self.__local_cache_dir}")
            if not os.path.exists(self.__local_cache_dir):
                os.makedirs(self.__local_cache_dir, exist_ok=True)
                save_config = {
                    "type": "graph-based",
                    "embedding_model_config": {
                        **self.__embedding_model_config,
                        "api_key": "...",
                    },
                }
                _save_as_json(save_config, f"{self.__local_cache_dir}/__config.json")

    def _make_project_graph(
        self,
        project: Project,
        project_itendifier: str,
    ) -> Any:
        # FIXME: do not directly call CodeToolkitBackend here
        from ..tools.code_toolkit import CodeToolkitBackend

        self.ilog(f"Building project graph for {project_itendifier}...")
        if self.__local_cache_dir is not None:
            path = f"{self.__local_cache_dir}/{project_itendifier}-project-graph.json"
            if os.path.isfile(path):
                self.ilog(f">>>> Loading project graph from {path}...")
                return CodeToolkitBackend.ProjectGraph.load(path)
            graph = CodeToolkitBackend.ProjectGraph.build(
                [project.get_src_sourcepath()],  # source roots
                [],  # class roots
            )
            self.ilog(f">>>> Saving project graph to {path}...")
            graph.save(path)
            return graph

    def _make_vector_store(
        self,
        project_graph: Any,
        project_itendifier: str,
    ) -> CodeSnippetVectorStore:
        self.ilog(f"Building vector store for {project_itendifier}...")
        if self.__local_cache_dir is not None:
            path = f"{self.__local_cache_dir}/{project_itendifier}-vector-store.json"
            if os.path.isfile(path):
                self.ilog(f">>>> Loading vector store from {path}...")
                return CodeSnippetVectorStore(
                    embedding_model_config=self.__embedding_model_config.copy(),
                    code_snippets=None,
                    load_filename=path,
                )
            store = CodeSnippetVectorStore(
                embedding_model_config=self.__embedding_model_config.copy(),
                code_snippets=self._collect_code_snippets(project_graph),
            )
            self.ilog(f">>>> Saving vector store to {path}...")
            store.save(path)
            return store

    def _collect_code_snippets(self, project_graph: Any) -> List[Dict[str, Any]]:
        self.ilog(f"Collecting code snippets from project graph...")
        code_snippets = []

        for node in tqdm.tqdm(
            project_graph.getFieldsNodes().values(),
            desc="Collecting Fields",
        ):
            code_snippets.append(
                {
                    "type": "fields",
                    "code": str(node.code),
                    "path": str(node.path),
                    "qualified_name": str(node.qualifiedName),
                    "file_path": str(node.filePath),
                    "_j_obj": {
                        "type": str(node.type),
                        "detailedType": str(node.detailedType),
                        "path": str(node.path),
                        "qualifiedName": str(node.qualifiedName),
                        "filePath": str(node.filePath),
                        "code": str(node.code),
                    },
                }
            )

        for node in tqdm.tqdm(
            project_graph.getMethodNodes().values(),
            desc="Collecting Methods",
        ):
            code_snippets.append(
                {
                    "type": "method",
                    "code": str(node.code),
                    "path": str(node.path),
                    "qualified_name": str(node.qualifiedName),
                    "file_path": str(node.filePath),
                    "_j_obj": {
                        "type": str(node.type),
                        "detailedType": str(node.detailedType),
                        "path": str(node.path),
                        "qualifiedName": str(node.qualifiedName),
                        "filePath": str(node.filePath),
                        "code": str(node.code),
                    },
                }
            )

        return code_snippets

    def _retrieve_subgraphs(
        self,
        focal_node: Any,
        query: str,
        project_graph: Any,
        vector_store: CodeSnippetVectorStore,
        strategy: str,
        num_retrieved_subgraphs: int,
    ):
        del query, vector_store  # unused now

        max_depth = int(project_graph.getMaxBFSDepth(focal_node))
        depth_list = {
            "only_max_subgraph": lambda M, D: [D] * (M + 1),
            "contextual_and_similarity": lambda M, D: [D] * (M + 1),
        }[strategy](M=num_retrieved_subgraphs - 1, D=max_depth)
        assert len(depth_list) == num_retrieved_subgraphs

        subgraphs = []
        last_subgraph_nodes = None
        for i, depth in enumerate(depth_list):
            nodes = list(project_graph.getSubgraphAround(focal_node, depth, True))
            num_nodes = len(nodes)

            if strategy.endswith("_expansion") and i > 0:
                assert last_subgraph_nodes is not None
                # assert len(last_subgraph_nodes) > 0
                last_subgraph_node_keys = {
                    (str(n.type), str(n.qualifiedName)) for n in last_subgraph_nodes
                }
                nodes = [
                    n
                    for n in nodes
                    if (str(n.type), str(n.qualifiedName))
                    not in last_subgraph_node_keys
                ]
                # assert len(nodes) < num_nodes
                assert len(nodes) + len(last_subgraph_nodes) == num_nodes
            del num_nodes

            last_subgraph_nodes = nodes
            subgraph = self.Subgraph(name=f"subgraph-focalnode-d{depth}", nodes=nodes)
            subgraphs.append(subgraph)

        if _is_debug_mode():
            self.ilog("Retrieving subgraphs...")
            self.ilog(f">>>> max bfs depth: {max_depth}")
            self.ilog(f">>>> bfs depth list: {depth_list}")
            self.ilog(f">>>> subgraphs l(nodes): {[len(g.nodes) for g in subgraphs]}")

        return subgraphs

    def _retrieve_code_snippets_in_subgraph(
        self,
        focal_node: Any,
        subgraph: Subgraph,
        query: str,
        retrieved: List[Tuple[str, List[CodeSnippet]]],
        project_graph: Any,
        vector_store: CodeSnippetVectorStore,
        num_retrieved_fields_nodes: int,
        num_retrieved_method_nodes: int,
    ) -> List[CodeSnippet]:
        from langchain_core.documents import Document

        del project_graph  # unused now

        assert str(focal_node.type) == "method"

        def _extract_retrieved_node_qualnames(type: str) -> set:
            qualnames = set()
            for _, cs_list in retrieved:
                for cs in cs_list:
                    cs_ty = str(cs.langchain_doc["metadata"]["type"])
                    assert cs_ty == cs.type and cs_ty in ["fields", "method"]
                    cs_qualname = str(cs.langchain_doc["metadata"]["qualified_name"])
                    if cs_ty == type:
                        # assert cs_qualname not in qualnames
                        qualnames.add(cs_qualname)
            return qualnames

        retrieved_fields_node_qualnames = _extract_retrieved_node_qualnames("fields")
        retrieved_method_node_qualnames = _extract_retrieved_node_qualnames("method")

        fields_node_qualnames = {
            str(n.qualifiedName)
            for n in subgraph.nodes
            if str(n.type) == "fields"
            # exclude retrieved
            if str(n.qualifiedName) not in retrieved_fields_node_qualnames
        }
        method_node_qualnames = {
            str(n.qualifiedName)
            for n in subgraph.nodes
            if str(n.type) == "method"
            # exclude retrieved
            and str(n.qualifiedName) not in retrieved_method_node_qualnames
            # exclude focal_node
            and str(n.qualifiedName) != str(focal_node.qualifiedName)
        }

        results: List[CodeSnippet] = []

        # 1st: retrieve (fields)
        def _fields_filter(doc) -> bool:
            return doc.metadata["qualified_name"] in fields_node_qualnames

        fields_docs = vector_store.search(
            query=query,
            filter=_fields_filter,
            k=num_retrieved_fields_nodes,
        )
        if len(fields_docs) < num_retrieved_fields_nodes:
            for i in range(num_retrieved_fields_nodes - len(fields_docs)):
                fields_docs.append(
                    Document(
                        page_content="N/A",
                        metadata={
                            "type": "fields",
                            "path": "N/A",
                            "file_path": "/N/A",
                            "qualified_name": f"f-N/A-{i}",
                        },
                    )
                )
        assert len(fields_docs) == num_retrieved_fields_nodes
        results.extend(
            CodeSnippet(
                type="fields",
                path=doc.metadata["path"],
                content=doc.page_content,
                file_path=doc.metadata.get("file_path", None),
                langchain_doc=doc.model_dump(mode="json"),
            )
            for doc in fields_docs
        )
        del _fields_filter, fields_docs

        # 2nd: retrieve (method)s
        def _method_filter(doc) -> bool:
            return doc.metadata["qualified_name"] in method_node_qualnames

        method_docs = vector_store.search(
            query=query,
            filter=_method_filter,
            k=num_retrieved_method_nodes,
        )
        if len(method_docs) < num_retrieved_method_nodes:
            for i in range(num_retrieved_method_nodes - len(method_docs)):
                method_docs.append(
                    Document(
                        page_content="N/A",
                        metadata={
                            "type": "method",
                            "path": "N/A",
                            "file_path": "/N/A",
                            "qualified_name": f"m-N/A-{i}",
                        },
                    )
                )
        assert len(method_docs) == num_retrieved_method_nodes
        results.extend(
            CodeSnippet(
                type="method",
                path=doc.metadata["path"],
                content=doc.page_content,
                file_path=doc.metadata.get("file_path", None),
                langchain_doc=doc.model_dump(mode="json"),
            )
            for doc in method_docs
        )
        del _method_filter, method_docs

        assert len(results) == num_retrieved_fields_nodes + num_retrieved_method_nodes
        return results

    def _rerank_code_snippets(
        self,
        rerank_prompt: str,
        code_snippets: List[CodeSnippet],
        project_graph: Any,
        vector_store: CodeSnippetVectorStore,
        num_rerank_tries_per_retrieved_node: int,
    ) -> List[CodeSnippet]:
        import numpy as np

        is_small_llm = any(
            m in self.__rerank_model.name.lower()
            for m in {
                # "qwen2.5-coder-32b-instruct",
                "qwen2.5-coder-14b-instruct",
                "deepseek-coder-33b-instruct",
                "deepseek-coder-v2-lite-instruct",
            }
        )

        del project_graph, vector_store  # unused now
        TOP_LOGPROBS = 20  # Set the proper value based on API limits
        original_indexes = [cs.additional_info["index_in_all"] for cs in code_snippets]

        for cs in code_snippets:
            if _is_debug_mode():
                cs_index = cs.additional_info["index_in_all"]
                self.ilog(f"Scoring code snippet {cs_index}...")

            cs_scores = []
            for i in range(num_rerank_tries_per_retrieved_node):
                N_TRY = 20
                for jj in range(N_TRY):
                    try:
                        r = self.__rerank_model.ask(
                            messages=[
                                Message.user(
                                    rerank_prompt.format(
                                        code_snippet=cs.content,
                                        output_format=(
                                            """\
Your output should JUST be the boolean true or false. Output 'true' if the provided code snippet is helpful, otherwise output 'false'.
Respond with just one word, the 'true' or 'false'. You must output the word 'true', or the word 'false', nothing else."""
                                            + (
                                                """\

Note that, the output should be lowercase: 'true' or 'false'.
Note that, the output should be lowercase: 'true' or 'false'.
Note that, the output should be lowercase: 'true' or 'false'.
Any other output is not allowed, except for the lowercase 'true' or 'false'.
Any other output is not allowed, except for the lowercase 'true' or 'false'.
Any other output is not allowed, except for the lowercase 'true' or 'false'.
"""
                                                if is_small_llm
                                                else ""
                                            )
                                        ),
                                    ),
                                )
                            ],
                            logprobs=True,
                            top_logprobs=TOP_LOGPROBS,
                            use_official_config=True,
                        )
                        assert r.logprobs is not None
                        logprobs_content = r.logprobs["content"]
                        if "llama-3.3-70b-instruct" in self.__rerank_model.name.lower():
                            self.wlog(f"Removing '<|eot_id|>' ... (llama 3.3)")
                            logprobs_content = logprobs_content[:-1]
                        if len(logprobs_content) != 1:
                            self.wlog(f"Response: {r.content}")
                            assert is_small_llm or (
                                "qwen2.5-coder-32b-instruct"
                                in self.__rerank_model.name.lower()
                            )
                            self.wlog(
                                f'len(r.logprobs["content"]): {len(logprobs_content)}'
                            )
                            self.wlog(f'r.logprobs["content"]: {logprobs_content}')
                            logprobs_content = [
                                next(
                                    c
                                    for c in logprobs_content
                                    if c["token"].strip().lower() in ["true", "false"]
                                )
                            ]
                        assert len(logprobs_content) == 1
                        assert len(logprobs_content[0]["top_logprobs"]) == TOP_LOGPROBS

                        top_logprobs = logprobs_content[0]["top_logprobs"]
                        if not is_small_llm:
                            true_token_logprob = next(
                                e["logprob"]
                                for e in top_logprobs
                                if e["token"] == "true"
                            )
                            false_token_logprob = next(
                                e["logprob"]
                                for e in top_logprobs
                                if e["token"] == "false"
                            )
                        else:
                            true_token_logprob = next(
                                e["logprob"]
                                for e in top_logprobs
                                if e["token"].strip() == "true"
                            )
                            false_token_logprob = next(
                                e["logprob"]
                                for e in top_logprobs
                                if e["token"].strip() == "false"
                            )
                        break
                    except Exception as e:
                        if _is_debug_mode():
                            __import__("traceback").print_exc()
                            self.wlog(f"Error: {e}")
                            self.wlog(f"Retrying {jj+1}...")
                        if jj == N_TRY - 1:
                            if (
                                "llama-3.3-70b-instruct"
                                in self.__rerank_model.name.lower()
                            ):
                                self.wlog(
                                    "Rerank model, llama-3.3-70b-instruct, is not working, skip"
                                )
                                raise SkipException(
                                    "rerank model (llama-3.3-70b-instruct) is not working"
                                )
                            raise e
                true_prob = np.exp(true_token_logprob)
                false_prob = np.exp(false_token_logprob)
                fixed_true_prob = true_prob / (true_prob + false_prob)
                cs_scores.append(fixed_true_prob)

            if _is_debug_mode():
                self.ilog(f">>>> Scores: {cs_scores}")
                self.ilog(f">>>> Avg Score: {sum(cs_scores) / len(cs_scores)}")

            cs.additional_info["scores"] = cs_scores
            cs.additional_info["avg_score"] = sum(cs_scores) / len(cs_scores)

        code_snippets = sorted(
            code_snippets,
            key=lambda cs: cs.additional_info["avg_score"],
            reverse=True,
        )
        reranked_indexes = [cs.additional_info["index_in_all"] for cs in code_snippets]

        if _is_debug_mode():
            self.ilog("Reranked code snippets")
            self.ilog(f">>>>> Rerank Prompt: `{rerank_prompt.splitlines()[0]}...`")
            self.ilog(f">>>>> Voters: {num_rerank_tries_per_retrieved_node}")
            self.ilog(f">>>>> Original Indexes: {original_indexes}")
            self.ilog(f">>>>> Reranked Indexes: {reranked_indexes}")

        return code_snippets

    def _group_code_snippets(
        self,
        code_snippets: List[CodeSnippet],
        num_group_of_retrieved: int,
        num_retrieved_nodes_per_group: int,
    ) -> List[Tuple[str, List[CodeSnippet]]]:
        num_left = num_group_of_retrieved * num_retrieved_nodes_per_group
        if len(code_snippets) < num_left:
            code_snippets.extend(
                CodeSnippet(
                    type="N/A",
                    path="N/A",
                    content="N/A",
                    file_path=None,
                    langchain_doc=None,
                    additional_info={"na": True},
                )
                for _ in range(num_left - len(code_snippets))
            )
        code_snippets = code_snippets[:num_left]  # top num_left
        assert num_left == len(code_snippets)

        return [
            (f"group-{i}", chunk)
            for i, chunk in enumerate(
                _split_list_as_chunks(
                    code_snippets,
                    chunk_size=num_retrieved_nodes_per_group,
                )
            )
        ]

    def _ask_impl(self, input: "Input") -> "Output":
        project = input.project
        project_itendifier = project.get_itendifier(ignore_project_version=False)
        project_graph = self._make_project_graph(project, project_itendifier)
        vector_store = self._make_vector_store(project_graph, project_itendifier)

        if self.__ratio_of_retrieved_fields_nodes == "auto":
            total_num_fields_nodes = len(project_graph.getFieldsNodes())
            total_num_method_nodes = len(project_graph.getMethodNodes())
            ratio_of_retrieved_fields_nodes = total_num_fields_nodes / (
                total_num_fields_nodes + total_num_method_nodes
            )
            num_retrieved_fields_nodes = int(
                self.__num_retrieved_nodes_per_subgraph
                * ratio_of_retrieved_fields_nodes
            )
            num_retrieved_method_nodes = (
                self.__num_retrieved_nodes_per_subgraph - num_retrieved_fields_nodes
            )
            if num_retrieved_fields_nodes == 0 and num_retrieved_method_nodes > 1:
                num_retrieved_fields_nodes += 1
                num_retrieved_method_nodes -= 1
            elif num_retrieved_method_nodes == 0 and num_retrieved_fields_nodes > 1:
                num_retrieved_method_nodes += 1
                num_retrieved_fields_nodes -= 1
            self.ilog(f"Auto calc fs/(fs+md): {ratio_of_retrieved_fields_nodes}")
            self.ilog(f">>>> total_num_fields_nodes: {total_num_fields_nodes}")
            self.ilog(f">>>> total_num_method_nodes: {total_num_method_nodes}")
            self.ilog(f">>>> num_retrieved_fields_nodes: {num_retrieved_fields_nodes}")
            self.ilog(f">>>> num_retrieved_method_nodes: {num_retrieved_method_nodes}")
            del total_num_fields_nodes, total_num_method_nodes
            del ratio_of_retrieved_fields_nodes
        else:
            num_retrieved_fields_nodes = self.__num_retrieved_fields_nodes_per_subgraph
            num_retrieved_method_nodes = self.__num_retrieved_method_nodes_per_subgraph

        focal_node = project_graph.getMethodNode(input.focal_method_qualified_name)
        if focal_node == None:
            return self.Output(
                project_name=project.get_itendifier(ignore_project_version=False),
                focal_method_qualified_name=input.focal_method_qualified_name,
                query=input.query,
                rerank_prompt=input.rerank_prompt,
                unranked_code_snippets=[],
                code_snippets=[("N/A", [])] * self.__num_group_of_retrieved,
            )

        query = input.query
        rerank_prompt = input.rerank_prompt

        self.ilog(f"Retrieving code snippets ...")
        self.ilog(f">> Query: `{query.splitlines()[0]}...`")
        self.ilog(f">> Focal Node: `{focal_node.type}@{focal_node.qualifiedName}`")

        # 1st: retrieve subgraphs
        self.ilog(f">> Retrieving subgraphs...")
        subgraphs = self._retrieve_subgraphs(
            focal_node=focal_node,
            query=query,
            project_graph=project_graph,
            vector_store=vector_store,
            strategy=self.__subgraph_selection_strategy,
            num_retrieved_subgraphs=self.__num_retrieved_subgraphs,
        )
        assert len(subgraphs) == self.__num_retrieved_subgraphs

        # 2nd: retrieve code snippets in each subgraph
        self.ilog(f">> Retrieving code snippets in each subgraph...")
        retrieved_from_subgraphs = []
        for i, subgraph in enumerate(subgraphs):
            self.ilog(f">>> Retrieving code snippets -- subgraph `{subgraph.name}`")
            retrieved_from_subgraphs.append(
                (
                    subgraph.name,
                    self._retrieve_code_snippets_in_subgraph(
                        focal_node=focal_node,
                        subgraph=subgraph,
                        query=query,
                        retrieved=retrieved_from_subgraphs,
                        project_graph=project_graph,
                        vector_store=vector_store,
                        num_retrieved_fields_nodes=num_retrieved_fields_nodes,
                        num_retrieved_method_nodes=num_retrieved_method_nodes,
                    ),
                )
            )
        assert len(retrieved_from_subgraphs) == self.__num_retrieved_subgraphs
        assert all(
            len(code_snippets) == self.__num_retrieved_nodes_per_subgraph
            for _, code_snippets in retrieved_from_subgraphs
        )

        # 2.x: retrieve contextual nodes
        if self.__subgraph_selection_strategy == "contextual_and_similarity":
            num_retrieved_contextual_nodes = self.__num_contextual_retrieved_nodes
            assert (
                num_retrieved_fields_nodes + num_retrieved_method_nodes
                == num_retrieved_contextual_nodes
            )
            self.ilog(f">> [!!!CONTEXTUAL!!!] Retrieving contextual nodes...")
            contextual_nodes = project_graph.getNodesFromEgoGraph(
                focal_node,  # ego
                num_retrieved_fields_nodes,  # numFieldsNodes
                num_retrieved_method_nodes,  # numMethodNodes
            )
            self.ilog(f">>> Retrieved ctx nodes (ori): {len(contextual_nodes)}")
            if len(contextual_nodes) > num_retrieved_contextual_nodes:
                contextual_nodes = contextual_nodes[:num_retrieved_contextual_nodes]
            self.ilog(f">>> Retrieved ctx nodes (f): {len(contextual_nodes)}")
            if _is_debug_mode():
                for i, node in enumerate(contextual_nodes):
                    self.ilog(
                        f">>>> Node({i}): {node.node.qualifiedName} (d={node.depth})"
                    )
                    self.ilog(f">>>>> Path: {node.forwardRelationshipsString}")

        # 3rd: merge code snippets
        self.ilog(f">> Merging code snippets...")
        flattened_retrieved = []
        index_in_all = 0
        visited = set()
        if self.__subgraph_selection_strategy == "contextual_and_similarity":
            for i, contextual_node in enumerate(contextual_nodes, start=index_in_all):
                assert str(contextual_node.node.qualifiedName) not in visited
                visited.add(str(contextual_node.node.qualifiedName))
                code_snippet = CodeSnippet(
                    type=str(contextual_node.node.type),
                    path=str(contextual_node.node.path),
                    content=str(contextual_node.node.code),
                    file_path=str(contextual_node.node.filePath),
                    langchain_doc=None,
                    additional_info={
                        "index_in_all": i,
                        "from": "contextual",
                        "type": str(contextual_node.node.type),
                        "detailedType": str(contextual_node.node.detailedType),
                        "path": str(contextual_node.node.path),
                        "file_path": str(contextual_node.node.filePath),
                        "qualified_name": str(contextual_node.node.qualifiedName),
                        "depth": int(contextual_node.depth),
                        "forward_relationships": str(
                            contextual_node.forwardRelationshipsString
                        ),
                        "path_for_rerank": str(
                            contextual_node.forwardRelationshipsString
                        ),
                    },
                )
                flattened_retrieved.append(code_snippet)
                index_in_all += 1
        for subgraph_name, code_snippets in retrieved_from_subgraphs:
            for i, code_snippet in enumerate(code_snippets, start=index_in_all):
                qualified_name = str(
                    code_snippet.langchain_doc["metadata"]["qualified_name"]
                )
                if qualified_name in visited:
                    self.ilog(f">>>>> Found a repeated node: {qualified_name}")
                    continue
                visited.add(qualified_name)
                code_snippet.additional_info["index_in_all"] = index_in_all
                code_snippet.additional_info["from"] = "subgraph"
                code_snippet.additional_info["subgraph"] = subgraph_name
                code_snippet.additional_info["index_in_retrieved_from_subgraph"] = i
                flattened_retrieved.append(code_snippet)
                index_in_all += 1
        self.ilog(f">>> Merged code snippets: {len(flattened_retrieved)}")

        # 4th: rerank code snippets
        unranked_code_snippets = flattened_retrieved
        if (
            "{code_snippet}" not in rerank_prompt
            or "{output_format}" not in rerank_prompt
        ):
            raise ValueError(
                "Rerank prompt must contain '{code_snippet}' and '{output_format}' as placeholders. "
                + f"Got ```{rerank_prompt}```"
            )
        self.ilog(f">>>> Reranking code snippets...")
        flattened_retrieved = self._rerank_code_snippets(
            rerank_prompt=rerank_prompt,
            code_snippets=unranked_code_snippets,
            project_graph=project_graph,
            vector_store=vector_store,
            num_rerank_tries_per_retrieved_node=self.__num_rerank_tries_per_retrieved_node,
        )

        # 5th: group code snippets
        self.ilog(f">>>> Grouping code snippets...")
        grouped_retrieved = self._group_code_snippets(
            code_snippets=flattened_retrieved,
            num_group_of_retrieved=self.__num_group_of_retrieved,
            num_retrieved_nodes_per_group=self.__num_retrieved_nodes_per_group,
        )
        assert len(grouped_retrieved) == self.__num_group_of_retrieved
        assert all(
            len(code_snippets) == self.__num_retrieved_nodes_per_group
            for _, code_snippets in grouped_retrieved
        )

        return self.Output(
            # input
            project_name=input.project.get_itendifier(ignore_project_version=False),
            focal_method_qualified_name=input.focal_method_qualified_name,
            query=input.query,
            rerank_prompt=input.rerank_prompt,
            # output
            unranked_code_snippets=unranked_code_snippets,
            code_snippets=grouped_retrieved,
        )
