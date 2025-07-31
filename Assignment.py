from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

import os

pdf_path = "zero to one novel.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(pages)

os.environ["OPENAI_API_KEY"] = "key"  


embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(documents, embeddings)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})


llm = ChatOpenAI(model_name="gpt-3.5-turbo")
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

questions = [
    "What does Peter Thiel mean by 'going from 0 to 1'?",
    "What are the key takeaways from Chapter 1?",
    "What is Peter Thiel's view on competition?",
    "How does Thiel define secrets in Chapter 3?",
    "What kind of future does Thiel argue for?"
]

for q in questions:
    print(f"\n Question: {q}")
    result = rag_chain(q)
    print(" Answer:\n", result['result'])
