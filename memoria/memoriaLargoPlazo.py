import os

from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver


BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # la url donde está el fichero actual...
SQLITE_PATH = os.path.join(BASE_DIR, "memoria_largo_plazo.sqlite")
MODEL_NAME = "gemma4:e2b" # cogéis uno bueno...

def imprimir_ultima_respuesta(respuesta):
    if not respuesta["messages"]:
        return

    respuesta["messages"][-1].pretty_print()


def main():
    print("Memoria persistente con SQLite.")
    print(f"Base de datos: {SQLITE_PATH}")
    print("Usad SIEMPRE el mismo thread_id para recuperar el historial entre sesiones.\n")

    thread_id = input("thread_id del usuario/sesion: ").strip() or "defecto" 
    
    with SqliteSaver.from_conn_string(SQLITE_PATH) as checkpointer: # Nos conectamos a la base de datos

        modelo = ChatOllama(
            model=MODEL_NAME,
            temperature=0.5,
        )

        agente = create_agent(
            modelo,
            tools=[],
            checkpointer=checkpointer,
        )

        while (prompt := input("> ")) != "end":
            for paso in agente.stream({
                "messages": [
                    HumanMessage(prompt)
                ]
            }, stream_mode="values",
            config={"configurable": {"thread_id": thread_id}
            }):
                ultimo_mensaje = paso["messages"][-1]

                hayRazonamiento = ""
                if hasattr(ultimo_mensaje, "additional_kwargs"): # sí, asi de escondido está el razonamiento
                    hayRazonamiento = ultimo_mensaje.additional_kwargs.get("reasoning_content", "")

                if hayRazonamiento:
                    print("\n=== PENSANDO ===")
                    print(hayRazonamiento)

                print("\n=== MENSAJE ===")
                if not isinstance(ultimo_mensaje, HumanMessage): # para que no me repita dos veces el msg del user
                    ultimo_mensaje.pretty_print()

if __name__ == "__main__":
    main()
