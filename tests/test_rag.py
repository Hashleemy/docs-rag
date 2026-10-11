from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.rag import format_context, message_text


def test_message_text_handles_plain_and_block_content():
    assert message_text(AIMessage(content="plain answer")) == "plain answer"
    blocks = [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "answer [1]"}]
    assert message_text(AIMessage(content=blocks)) == "answer [1]"


def test_format_context_numbers_excerpts_with_source_and_page():
    docs = [
        Document(page_content="first", metadata={"source": "a.pdf", "page": 3}),
        Document(page_content="second", metadata={"source": "b.md"}),
    ]
    text = format_context(docs)
    assert text.startswith("[1] (a.pdf, page 3)\nfirst")
    assert "[2] (b.md)\nsecond" in text
