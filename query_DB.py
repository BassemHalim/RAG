import argparse

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chromaDB"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Your text query")
    args = parser.parse_args()
    query = args.query_text

    embedding_func = HuggingFaceEmbeddings()
    DB = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_func)

    res = DB.similarity_search_with_relevance_scores(query=query, k=4)
    if len(res) == 0 or res[0][1] > 0.7:
        print("Unable to find matching result")
        return

    context_text = "\n\n--\n\n".join(
        [f"{score}:\n{doc.metadata}\n{doc.page_content}" for doc, score in res]
    )
    print(context_text)


if __name__ == "__main__":
    main()
