from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import os

# 1. Load document
loader = TextLoader("example.txt")
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Set up a prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use the following context to answer the question:
{context}

Question: {question}
""")

# 4. Set up the LLMChain
llm = ChatOpenAI(temperature=0)
chain = LLMChain(llm=llm, prompt=prompt)

# 5. Run example question
context = chunks[0].page_content
question = "What is LangChain?"

response = chain.run({"context": context, "question": question})
print("Answer:", response)
