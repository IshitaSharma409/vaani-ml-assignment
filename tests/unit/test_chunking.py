from langchain_core.documents import Document

from relay.retrieval.chunking import split_pages


def test_chunks_keep_source_page_and_stable_ids():
    page = Document(page_content=("Attention is all you need. " * 100), metadata={
        "paper_id": "1706.03762", "arxiv_version": "7", "pdf_page_number": 4,
    })
    first = split_pages([page], 180, 30)
    second = split_pages([page], 180, 30)
    assert len(first) > 1
    assert [d.metadata["chunk_id"] for d in first] == [d.metadata["chunk_id"] for d in second]
    assert all(d.metadata["pdf_page_number"] == 4 for d in first)
    assert all(len(d.page_content) <= 180 for d in first)
