import streamlit as st
from langchain_together import ChatTogether
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ChatMessageHistory
import time
import pandas as pd
from io import StringIO
from PyPDF2 import PdfReader
from langchain_community.document_loaders import WebBaseLoader
from langchain_together import TogetherEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

############################## Functions ################################
max_tokens = 8192
api_key = "your api key"

# Function to truncate messages to fit within the token limit
def truncate_messages(messages, max_tokens):
    total_tokens = sum([len(msg.content.split()) for msg in messages])
    while total_tokens > max_tokens and messages:
        total_tokens -= len(messages.pop(0).content.split())
    return messages

def get_pdf_text(pdf_docs):
    text = ""
    if not isinstance(pdf_docs, list):
        pdf_docs = [pdf_docs]
    for pdf in pdf_docs:
        pdf.seek(0)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    if isinstance(text, list):
        text = " ".join(text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap = 500)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks, api_key):
    embeddings = TogetherEmbeddings(together_api_key = api_key)
    vector_store = FAISS.from_texts(chunks, embeddings)
    #vector_store.save_local("faiss_index")
    return vector_store

def llm_model(vector):
    llm = ChatTogether(together_api_key = api_key, model="meta-llama/Llama-3-8b-chat-hf", temperature=0.2)
    output_parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_template("""Answer the following question based on the provided context:
        <context>{context}</context>
        Question: {input}""")
    
    document_chain = create_stuff_documents_chain(llm, prompt, output_parser=output_parser)
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def chat(prompt_text):
    # React to user input
    if prompt_text:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt_text)
            st.session_state.demo_ephemeral_chat_history.add_user_message(prompt_text)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        st.session_state.demo_ephemeral_chat_history.messages = truncate_messages(st.session_state.demo_ephemeral_chat_history.messages, 8000)
        
        if "vector" in st.session_state:
            chain = llm_model(st.session_state.vector)
            response = chain.invoke({"input": prompt_text, "context": st.session_state.demo_ephemeral_chat_history.messages})

            # Ensure response is a string
            if isinstance(response, dict) and "answer" in response:
                response = response["answer"]
            elif isinstance(response, list):
                response = " ".join(response)
            else:
                response = str(response)
        else:
            response = "Please process a document first."

        st.session_state.demo_ephemeral_chat_history.add_ai_message(response)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    add_selectbox = st.sidebar.selectbox("What would you like to import?", ("PDF", "Web URL"))

    if add_selectbox == "PDF":
        with st.sidebar:
            uploaded_file = st.file_uploader("Choose a file", type="pdf", key="file_uploader")
            if st.button("Submit & Process", key="process_btn") and uploaded_file is not None:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(uploaded_file)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector = get_vector_store(text_chunks, api_key)
                    st.success("Done!")
    else:
        with st.sidebar:
            url = st.text_input("Enter Web URL")
            if st.button("Submit & Process", key="process_btn") and url is not None:
                with st.spinner("Processing..."):
                    loader = WebBaseLoader(url)
                    documents = loader.load()
                    raw_text = "\n".join([doc.page_content for doc in documents])
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector = get_vector_store(text_chunks, api_key)
                    time.sleep(5)
                    st.success("Done!")

    st.title("Chat Bot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello, I'm a chat bot. How can I help you?"}]
        st.session_state.demo_ephemeral_chat_history = ChatMessageHistory()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            st.markdown(
                """
                <style>
                    .st-emotion-cache-4oy321{
                        flex-direction: row-reverse;
                        text-align: left;
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )
    
    prompt_text = st.chat_input("Ask something")
    chat(prompt_text)

if __name__ == "__main__":
    main()