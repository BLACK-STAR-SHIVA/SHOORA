# SHOORA

Open Google Colab 
Download the Mistral Model and upload on Drive on same account from : https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF/tree/main
You can Load any Pdf related to the Field

## Loading the Google Drive 
from google.colab import drive
drive.mount("/content/drive")

## Installation
pip install langchain sentence-transformers chromadb llama-cpp-python langchain_community pypdf

## Installing Libraries
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA, LLMChain

## Importing Pdf
loader = PyPDFDirectoryLoader("/content/drive/MyDrive/Generative AI/Medical Bot")
docs = loader.load()

## Chunking
text_splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

## Embeddings
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

## Vector Store Creation
vectorstore = Chroma.from_documents(chunks, embeddings)
query = "who is at heart risk?"
search_results = vectorstore.similarity_search(query)
search results
retriever = vectorstore.as_retriever(search_kwargs={'k':5})
retriever.get_relevant_documents(query)

## LLM Model Loading
llm = LlamaCpp(
model_path="",
temperature=0.2,
max_tokens=2048,
top_p=1
)

## Use LLM and retriver and query, to generate final Response
template = """
<|context|>
you are medical Asistant that follows the instructions and generate the accurate response based on the query and the context provided.
Please be truthful and give direct answers.
<\s>
<|user|>
{query}
<\s>
<|assistant|>
"""

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
{"context";retriever, "query":RunnablePassthrough()}
| prompt
| llm
| StrOutputParser()
)

response=rag_chain.invoke(query)

response

import sys
--space--
while True:
user _input = input(f"Input query: ")
if user _input == 'exit':
print("Exiting...")
sys.exit()
if user _input=="":
continue
result = rag_chain.invoke(user_input)
print("Answer :", result)
