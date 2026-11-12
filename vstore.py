"""One import for the vector store. Picks ChromaDB (local) or pgvector (cloud) from DOCUMIND_BACKEND."""
import config

if config.BACKEND == "postgres":
    import store_pg as _impl
else:
    import store as _impl

DemoLimitError = _impl.DemoLimitError
index_document = _impl.index_document
search = _impl.search
delete_document = _impl.delete_document
count_chunks = _impl.count_chunks
