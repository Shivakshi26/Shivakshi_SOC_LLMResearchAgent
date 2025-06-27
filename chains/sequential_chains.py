from langchain_groq import ChatGroq
from dotenv import load_dotenv 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
)

prompt1 = PromptTemplate(
    template= ' Generate a detailed report on {topic}',
    input_variables= ['topic']
)
prompt2 = PromptTemplate( 
    template= ' Generate a 5 point summary from the following text \n {text}',
    input_variables= ['text']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser 

result = chain.invoke({'topic':'Math'})

print(result)

chain.get_graph().print_ascii()
