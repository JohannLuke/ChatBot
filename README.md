# ChatBot using Streamlit and LangChain

This project is a chatbot built using **Streamlit** and **LangChain**, supporting PDF and web-based document processing. The chatbot leverages **FAISS** for vector storage and **Meta-Llama-3-8B** for text generation.

## Features
- 📝 Accepts **PDF files** or **Web URLs** as input.
- 🔍 Extracts and processes text into **vector embeddings** using FAISS.
- 💬 Supports chat-based interactions with AI.
- 🔄 Maintains conversation history and truncates messages for efficiency.
- 💡  Built with Streamlit & LangChain

## Installation
Ensure you have Python installed (preferably **Python 3.8+**). Then, install the required dependencies:

```bash
pip install streamlit langchain langchain_together PyPDF2 pandas FAISS-cpu
```

## Usage
Run the chatbot with:

```bash
streamlit run chatbot.py
```

## API Key Setup
Replace "your_api_key" in chatbot.py with your actual API key for Together AI.

## How It Works
Upload a PDF file or enter a Web URL.

The document is processed and split into smaller chunks.

The chunks are converted into vector embeddings for retrieval.

The chatbot uses Llama-3-8B to generate responses based on retrieved context.

## Project Structure
```bash
📂 ChatBot
 ├── chatbot.py        # Main application file
 ├── README.md         # Project documentation
 ├── requirements.txt  # Required dependencies
```

## Future Improvements
🔧 Support for additional document formats (TXT, DOCX).

🚀 Improve retrieval and response accuracy.

🌍 Deploy as a web service.
