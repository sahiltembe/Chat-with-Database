from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
import streamlit as st 
from urllib.parse import quote_plus

def init_database(host: str, user: str, password: str, database: str ) -> SQLDatabase:
    encoded_password = quote_plus(password)
    db_uri = f"mysql+pymysql://{user}:{encoded_password}@{host}/{database}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template = """
        Your are a data analyst at a company. You are interacting with a user who is asking you question about the company's database.
        Based on the table schema below, write a sql query that would answer the user's question. Take the conversation history into account.

        <SCHEMA>{schema}</SCHEMA>

        Converation History: {chat_history}

        Write only the SQL query and nothing else. Do Not wrap the SQL query in any other text, not even backticks.

        For example:
        Question: Calculate the total revenue generated from pizza sales ?
        SQL Query: select round(sum(order_details.quantity * pizzas.price)) as total_sales from order_details join pizzas on pizzas.pizza_id = order_details.pizza_id;
        Question: Retrieve the total number of orders placed ?
        SQL Query: select count(order_id) as total_orders from orders;

        Your turn:
        Question:{question}
        SQL Query: 
        """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-0125-preview")

    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()


    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)

    template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database. 
        Based on the table schema below, question, sql query, and sql response, write a natural language response.

        <SCHEMA>{schema}</SCHEMA>

        Conversation History: {chat_history}  
        SQL Query: <SQL>{query}</SQL>  
        Question: {question}  
        SQL Response: {response}

        """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-0125-preview")

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke({
        "question":user_query,
        "chat_history":chat_history,
    })



if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage("Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

load_dotenv()

st.set_page_config(page_title="Chat With MYSQL", page_icon="🦜")
st.title("Chat With MYSQL")

with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using MySQL. Connect to the database and start chatting.")

    st.text_input("Host", value="localhost", key="host")
    st.text_input("User", value="root", key="user")
    st.text_input("Password", type="password", key="password")
    st.text_input("Database", value="pizzahut", key="database")

    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            try:
                db = init_database(
                    st.session_state.host,
                    st.session_state.user,
                    st.session_state.password,
                    st.session_state.database
                )
                st.session_state.db = db
                st.success("✅ Connected to database!")
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() !="":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("human"):
        st.markdown(user_query)

    with st.chat_message("ai"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)

    st.session_state.chat_history.append(AIMessage(content=response))

