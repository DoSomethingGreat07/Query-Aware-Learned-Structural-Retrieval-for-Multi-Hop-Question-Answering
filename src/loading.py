from datasets import load_dataset
from langchain_core.documents import Document


def load_hotpotqa_documents(split="train", max_samples=1000):
    """
    Load HotpotQA context paragraphs and deduplicate them by (title, text).

    Returns:
        List[Document]
    """
    dataset = load_dataset("hotpot_qa", "distractor", split=split)

    documents = []
    seen = set()

    max_samples = min(max_samples, len(dataset))

    for item in dataset.select(range(max_samples)):
        titles = item["context"]["title"]
        sentences_list = item["context"]["sentences"]

        for title, sentences in zip(titles, sentences_list):
            text = " ".join(sentences).strip()

            if not text:
                continue

            key = (title.strip(), text)
            if key in seen:
                continue
            seen.add(key)

            doc = Document(
                page_content=text,
                metadata={
                    "title": title,
                    "source": "HotpotQA",
                },
            )
            documents.append(doc)

    return documents


if __name__ == "__main__":
    docs = load_hotpotqa_documents(split="train", max_samples=1000)

    print("Total unique documents loaded:", len(docs))
    print("\nSample metadata:")
    print(docs[0].metadata)

    print("\nSample text:")
    print(docs[0].page_content[:500])