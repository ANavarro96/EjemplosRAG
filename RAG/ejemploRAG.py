from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage, SystemMessage
from dataclasses import dataclass


@dataclass
class Contexto:
    retriever: VectorStoreRetriever

# =========================
# CONFIG
# =========================
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "streamers"



# Modelo generativo en Ollama para responder
LLM_MODEL = "gemma4:e2b"  


def crear_embeddings():

    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large", # El modelo LLM a usar
        base_url="http://localhost:11434", # Esta es la URL de Ollama (local)
    )
    return embeddings


def crear_retriever(vectorstore: Chroma):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    return retriever



@tool
def obtenerReceta(query: str, contexto:ToolRuntime[Contexto]):
    """
    Llama a esta herramienta siempre que el usuario quiera obtener información sobre el xokas
    Como vas a acceder a una base de datos vectorial, usa la query para realizar una búsqueda semántica
    args:
        query: La pregunta del usuario
    """

    vectorstore = contexto.context.retriever

    return vectorstore.invoke(query)

    


# =========================
# 9) MAIN
# =========================
def main():


    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=crear_embeddings(),
    )

    print("..Chromadb listp...")

    retriever = crear_retriever(vectorstore)

    print("...Chromadb integrado...")

    print("RAG listo. Escribe una pregunta sobre el PDF.")
    print("Escribe 'salir' para terminar.\n")



    modelo = ChatOllama(model = "gemma4:e2b", reasoning=True)

    agente = create_agent(model=modelo, system_prompt='''
    Eres un asistente respetuoso y atento, que llama a herramientas.
    Vas a llamar a las herramientas cuando el usuario te pregunte sobre el xokas
    No inventes la salida
    No cambies de tema
    ''', tools=[obtenerReceta], context_schema=Contexto)

    while (prompt := input("> ")) != "end":
        for paso in agente.stream({
            "messages": [
                HumanMessage(prompt)
            ]
        }, stream_mode="values", context=Contexto(retriever=retriever)):
            ultimo_mensaje = paso["messages"][-1]

            hayRazonamiento = ""
            if hasattr(ultimo_mensaje, "additional_kwargs"): # sí, asi de escondido está el razonamiento
                hayRazonamiento = ultimo_mensaje.additional_kwargs.get("reasoning_content", "")

            if hayRazonamiento:
                print("\n=== PENSANDO ===")
                print(hayRazonamiento)

            print("\n=== MENSAJE ===")
            ultimo_mensaje.pretty_print()


if __name__ == "__main__":
    main()