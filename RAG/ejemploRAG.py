from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage, SystemMessage

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "streamers"


LLM_MODEL = "gemma4:e2b"  


def crear_embeddings():

    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large", # El modelo LLM a usar. Que sea el mismo con el que vectorizamos los documentos!
        base_url="http://localhost:11434", # Esta es la URL de Ollama (local)
    )
    return embeddings


def crear_retriever(vectorstore: Chroma):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    return retriever
 

def main():


    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=crear_embeddings(),
    )

    print("..Chromadb listp...")

    retriever = crear_retriever(vectorstore)

    print("...Chromadb integrado...")

    resultado = retriever.invoke("¿De dónde es elxokas?")

    for indice,documento in enumerate(resultado):
        print(f"documento num {indice} -> {documento}")
    """
    De cada documento sacamos su contenido (page_content) y sus metadatos, si es que hay.
    El retriever es una abstracción, si queremos sacar distancia/similitud, hay que usar
    el vectorstore.
    """

    resultado = vectorstore.similarity_search_with_score("¿De dónde es elxokas?", k = 4)

    for indice,documento in enumerate(resultado):
        print(f"documento num {indice} -> {documento}")

    

    


if __name__ == "__main__":
    main()