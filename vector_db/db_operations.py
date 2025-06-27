import uuid
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(dim)
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)


docs = [
    Document(page_content="First item about LangChain", metadata={"source": "note"}),
    Document(page_content="Another entry about FAISS operations", metadata={"source": "note"}),
    Document(page_content="This document will be deleted", metadata={"source": "temp"}),
]
ids = [str(uuid.uuid4()) for _ in docs]
vector_store.add_documents(documents=docs, ids=ids)


results = vector_store.similarity_search("FAISS", k=3)
for doc in results:
    print(f"→ {doc.page_content} [{doc.metadata}]")

vector_store.delete(ids=[ids[2]])

# Perform similarity search with metadata filtering
filtered = vector_store.similarity_search("LangChain", k=2, filter={"source": "note"})
for doc in filtered:
    print(f"[Filtered] → {doc.page_content} [{doc.metadata}]")

# Similarity search with scores
scored = vector_store.similarity_search_with_score("FAISS", k=2)
for doc, score in scored:
    print(f"[SIM={score:.3f}] → {doc.page_content} [{doc.metadata}]")

# Q&A using RetrievalQA chain
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    retriever=retriever
)
answer = qa_chain.run("What is FAISS used for?")
print("\n[Q&A Answer]\n", answer)

# Save & load the vector database
vector_store.save_local("faiss_db")
loaded_vs = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
print("Reloaded entries (simi search):", loaded_vs.similarity_search("LangChain", k=2))
