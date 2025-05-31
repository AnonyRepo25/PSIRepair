from os import getenv
from typing import Any, List
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from .rs_utils import _abstractmethod, _ilog, _split_list_as_chunks


def _make_embedding_model(embedding_model_config):
    def _import_openai_embeddings():
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings

    print_kwargs = {**embedding_model_config, "api_key": "..."}
    _ilog(f"Creating {embedding_model_config['type']} embed model, cfg={print_kwargs}")

    embedding_model_config = embedding_model_config.copy()
    emb_type = embedding_model_config.pop("type")
    local_cache_path = embedding_model_config.pop("local_cache_path")
    namespace = embedding_model_config.pop("namespace")

    base_url = embedding_model_config.get("base_url", None)
    if base_url and base_url.startswith("$"):
        if not (base_url := getenv(base_url[1:], None)):
            raise ValueError(f"Env var {base_url[1:]} is not set")
        embedding_model_config["base_url"] = base_url

    emb_cls = {
        "openai": _import_openai_embeddings,
    }[emb_type]()

    underlying_embeddings = emb_cls(**embedding_model_config)
    local_store = LocalFileStore(local_cache_path)
    return CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings,
        local_store,
        namespace=namespace,
    )


class VectorStore:
    @_abstractmethod
    def search(self, query: Any, k: int) -> List[Document]:
        pass


class CodeSnippetVectorStore(VectorStore):
    def __init__(
        self,
        embedding_model_config,
        code_snippets=None,
        load_filename=None,
        **kwargs,
    ):
        super().__init__()
        _ilog(f"Building code snippet vector store...")
        _ilog(f">>>> embeddings: {embedding_model_config['type']}")
        _ilog(f">>>> l(code_snippets): {len(code_snippets) if code_snippets else 'NA'}")
        self.__embedding_model = _make_embedding_model(embedding_model_config)
        if load_filename is not None:
            self.__base_vector_store = InMemoryVectorStore.load(
                path=load_filename,
                embedding=self.__embedding_model,
            )
        else:
            self.__base_vector_store = InMemoryVectorStore(
                embedding=self.__embedding_model
            )
        if code_snippets is not None:
            code_snippets_parts = list(
                _split_list_as_chunks(code_snippets, chunk_size=100)
            )
            assert sum(len(p) for p in code_snippets_parts) == len(code_snippets)
            added_count = 0
            total_steps = len(code_snippets_parts)
            for i, cs_l in enumerate(code_snippets_parts, start=1):
                _ilog(
                    f">>>> [{i}/{total_steps}] Added {added_count} cs to vector store"
                )
                self.__base_vector_store.add_texts(
                    texts=[cs.pop("code") for cs in cs_l],
                    metadatas=cs_l,
                )
                added_count += len(cs_l)
            assert added_count == len(code_snippets)
            _ilog(f">>>> Added {added_count} code snippets to vector store")

    def save(self, path):
        self.__base_vector_store.dump(path)

    def search(self, query, k, filter=None) -> List[Document]:
        return self.__base_vector_store.similarity_search(
            query=query, filter=filter, k=k
        )
