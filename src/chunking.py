from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loading import load_hotpotqa_documents


def chunk_documents(
    split="train",
    max_samples=1000,
    chunk_size=300,
    chunk_overlap=50,
):
    docs = load_hotpotqa_documents(split=split, max_samples=max_samples)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    title_counts = defaultdict(int)

    for i, chunk in enumerate(chunks):
        title = chunk.metadata.get("title", "")
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_index_in_title"] = title_counts[title]
        title_counts[title] += 1

    return chunks


if __name__ == "__main__":
    chunks = chunk_documents()

    print("Total chunks:", len(chunks))
    print("\nSample chunk metadata:")
    print(chunks[0].metadata)

    print("\nSample chunk text:")
    print(chunks[0].page_content)