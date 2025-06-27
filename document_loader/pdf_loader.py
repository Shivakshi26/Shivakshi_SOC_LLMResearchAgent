from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader

loader= PyPDFLoader("document_loader\paper.pdf")

loader2 = UnstructuredPDFLoader("document_loader\paper.pdf")
docs = loader2.load()


print(docs[0].page_content)