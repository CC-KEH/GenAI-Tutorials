import os
import getpass
from typing import Sequence
from typing_extensions import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    trim_messages,
    BaseMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langgraph.graph.message import add_messages

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass(
        "Enter your Google AI API key: ")


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


class Model:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        self.trimmer = trim_messages(
            max_tokens=65,
            strategy="last",
            token_counter=self.model,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )
        self.workflow = StateGraph(state_schema=State)
        self.workflow.add_edge(START, "model")
        self.workflow.add_node("model", self.call_model)

        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)

    def call_model(self, state: State):
        if not state["messages"] or len(state["messages"]) == 1:
            state["messages"] = self.chat_message_history.messages + \
                state["messages"]
        trimmed_messages = self.trimmer.invoke(state["messages"])
        prompt = self.prompt_template.invoke(
            {"messages": trimmed_messages, "language": state["language"]}
        )
        response = self.model.invoke(prompt)

        self.chat_message_history.add_user_message(prompt.messages[-1].content)
        self.chat_message_history.add_ai_message(response.content)

        return {"messages": [response]}

    def invoke(self, query, config, chat_message_history):
        self.chat_message_history = chat_message_history
        input_messages = [HumanMessage(query)]
        response = self.app.invoke(
            {"messages": input_messages, "language": "en"}, config
        )
        return response["messages"][-1].content


if __name__ == "__main__":
    model = Model()
    no_of_users = input("Enter the number of users: ")

    # Create memory with initial queries
    for i in range(int(no_of_users)):
        query = input(f"Enter your query 1 for user{i+1}: ")
        config = {"configurable": {"thread_id": "user" + str(i)}}
        chat_message_history = MongoDBChatMessageHistory(
            session_id="user" + str(i),
            connection_string="mongodb://localhost:27017",
            database_name="my_db",
            collection_name="chat_histories",
        )
        print(model.invoke(query, config, chat_message_history))

    # Check memory with new queries
    for i in range(int(no_of_users)):
        query = input(f"Enter your query 2 for user{i+1}: ")
        config = {"configurable": {"thread_id": "user" + str(i)}}
        chat_message_history = MongoDBChatMessageHistory(
            session_id="user" + str(i),
            connection_string="mongodb://localhost:27017",
            database_name="my_db",
            collection_name="chat_histories",
        )
        print(model.invoke(query, config, chat_message_history))
