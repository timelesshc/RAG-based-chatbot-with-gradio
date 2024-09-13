# RAG-based-chatbot-with-gradio
Build your own Custom RAG Chatbot using Gradio, Langchain and ChatGPT

## Technologies Used
- Langchain
- RAG
- Bge Embedding & Reranker
- Gradio
- ChromaDB

## Features
- Process PDF and Word files and extract information for answering questions.
- Maintain chat history and provide detailed explanations.
- Generate responses using a Conversational Retrieval Chain.
- Return to casual chat if retriever could not find the answers in the database

## Prerequisites
Before running the ChatBot, ensure that you have the required dependencies installed. You can install them using the following command:
```
pip install -r requirements.txt
```
Input your OpenAI API key in the `.env` document

## Run the Code
Download bge-reranker-large and bge-larger-zh-v1.5 model and save to `BAAI` folder.

Run command:
```
python main.py
```
## License
This project is licensed under the [Apache License 2.0](https://github.com/timelesshc/RAG-based-chatbot-with-gradio/blob/main/LICENSE).
