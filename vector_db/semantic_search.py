from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
import faiss


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

dim = len(embeddings.embed_query("Hello world"))
index = faiss.IndexFlatL2(dim)
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

from langchain.core.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

loader = TextLoader("my_docs.txt")
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)


vector_store.add_documents(chunks)

query = "What's the summary of section about FAISS?"
results = vector_store.similarity_search(query, k=5)

for doc in results:
    print("---\n", doc.page_content)

vector_store.save_local("faiss_index")
vs2 = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

vector_store.delete()