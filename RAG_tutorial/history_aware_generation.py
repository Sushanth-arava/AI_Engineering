from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel
from config import config


load_dotenv()


embeddings = OpenAIEmbeddings(model=config.models.embedding_model)
db = Chroma(persist_directory=config.storage.persist_directory, embedding_function=embeddings)


model = ChatOpenAI(model=config.models.llm_model)


# **:  Pydantic models for structured LLM output
class RewrittenQuestion(BaseModel):
    rewritten_question: str


class RAGAnswer(BaseModel):
    answer: str
    sources: list[str]


# **:  Pydantic model for typed chat history
class ChatTurn(BaseModel):
    user: str
    assistant: str


class ConversationState(BaseModel):
    turns: list[ChatTurn] = []

    def to_langchain_messages(self) -> list:
        messages = []
        for turn in self.turns:
            messages.append(HumanMessage(content=turn.user))
            messages.append(AIMessage(content=turn.assistant))
        return messages


conversation = ConversationState()


def ask_question(user_question):

    print(f"User asked: {user_question}")

    history_messages = conversation.to_langchain_messages()

    if history_messages:
        rewrite_model = model.with_structured_output(RewrittenQuestion)
        messages = (
            [
                SystemMessage(
                    content="Given the chat history, rewrite the new question to be a standalone and searchable. Just return the rewritten question"
                )
            ]
            + history_messages
            + [HumanMessage(content=f"New question: {user_question}")]
        )
        result: RewrittenQuestion = rewrite_model.invoke(messages)
        search_question = result.rewritten_question
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # **:  Retrieves top k results
    retriever = db.as_retriever(search_kwargs={"k": config.retrieval.k})
    docs = retriever.invoke(search_question)

    combined_input = f"""Based on the knowledge from the documents, answer this query: {user_question}

    Documents: {chr(10).join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, detailed and helpful answer using only the information from these documents. If you can't find any relevant information, please say 'I couldn't find any info from these docs'.
    Also list the source filenames of documents you used in the sources field.
    """

    answer_model = model.with_structured_output(RAGAnswer)
    messages = (
        [
            SystemMessage(
                content="You are a helpful assistant that answers questions based on provided documents and conversation history."
            ),
        ]
        + history_messages
        + [HumanMessage(content=combined_input)]
    )
    result: RAGAnswer = answer_model.invoke(messages)

    # Store this turn in typed conversation history
    conversation.turns.append(ChatTurn(user=user_question, assistant=result.answer))

    print(f"Answer: {result.answer}")
    print(f"Sources: {result.sources}")
    return result


def start_chat():
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\n Your question: ")

        if question.lower() == "quit":
            print("Goodbye and Thank you")
            break

        ask_question(question)


if __name__ == "__main__":
    start_chat()
