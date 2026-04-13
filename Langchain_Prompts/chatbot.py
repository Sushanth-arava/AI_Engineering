from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()
chat_history = []
while True:
    user_input = input("You:  ")

    if user_input == "exit":
        break
    result = model.invoke(user_input)
    chat_history.append({"User": user_input, "AI": result.content})
    print("AI: ", result.content)

print(chat_history)
