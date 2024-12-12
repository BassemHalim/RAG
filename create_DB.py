from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "data/books"
CHROMA_PATH = "chromaDB"


def load_documents() -> List[Document]:
    loader = DirectoryLoader(DATA_PATH, "*.md")
    docs = loader.load()
    return docs


def chunk_docs(docs: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} into {len(chunks)} chunks.")
    return chunks
    # chunk = chunks[5]
    # print(chunk.page_content)
    # print(chunk.metadata)


def save_to_chroma(chunks: List[Document]):
    db = Chroma.from_documents(
        documents=chunks,
        embedding=HuggingFaceEmbeddings(),
        persist_directory=CHROMA_PATH,
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def main():
    docs = load_documents()
    chunks = chunk_docs(docs)
    save_to_chroma(chunks)


if __name__ == "__main__":
    main()
