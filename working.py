import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import  PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

import chromadb.api

chromadb.api.client.SharedSystemClient.clear_system_cache()


import os

embeddings = OllamaEmbeddings(model="llama3.2")
llm = ChatOllama(model="llama3.2")


st.title("Chat with your documents")
st.write("You can upload your own document and ask question on it...")

session_id = st.text_input("Session ID", value="default_session")
if "store" not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = []
    for file in uploaded_files:
        file_name = file.name
        with open(f"./temp/{file_name}.pdf","wb") as f:
            f.write(file.getvalue())
        
        loader = PyPDFLoader(f"./temp/{file_name}.pdf")
        doc = loader.load()
        documents.extend(doc)
    
    # split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # create a vector store 
    vector_db = Chroma.from_documents(splits, embeddings)
    retriever = vector_db.as_retriever()

    prompt_text = (
        "Given a chat history and latest user question"
        "which might reference context in the chat history"
        "formulate a standlong question which can be understood"
        "without the chat history, DO NOT answer the question"
        "just reformulate it if needed otherwise return it as is"
    )

    context_qna_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_qna_prompt)

    # answer question
    system_prompt = (
        "You are an assistant for question answering tasks"
        "Use the following pieces of retrieved context to answer the questions"
        "If you dont know the answer, just say that you dont know"
        "\n\n"
        "Context: {context}"
    )

    qna_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qna_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    user_input = st.text_input("Your question: ")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            { "input": user_input },
            config = {
                "configurable": {"session_id": session_id}
            }
        )

        # st.write(st.session_state.store)
        st.success(f"Assistant: {response['answer']}")
        # st.write("Chat history: ", session_history.messages)