from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

result = model.invoke("When did Gandhi die?")
print(result.content)
