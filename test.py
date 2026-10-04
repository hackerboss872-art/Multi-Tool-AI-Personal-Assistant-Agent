from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.4
)

user_input = input("Enter your query: ")

response = llm.invoke(user_input)

print("\nAI:", response.content)