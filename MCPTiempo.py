import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain.tools import tool
import json
from textwrap import indent
# lanzar pip install mcp_weather_server


# Esta función pinta de forma bonita las herramientas que se descargan de un mcp
def pretty_tool(tool):
    print(f"\n🛠️ TOOL: {tool.name}")
    print("-" * 50)

    print("📄 Descripción:")
    print(indent(tool.description.strip(), "  "))

    print("\n📥 Argumentos (JSON Schema):")
    try:
        schema = tool.args_schema
        print(indent(json.dumps(schema, indent=2, ensure_ascii=False), "  "))
    except Exception:
        print("  No disponible")

    print("\n📌 Campos requeridos:")
    try:
        required = tool.args_schema.get("required", [])
        print(f"  {required}")
    except Exception:
        print("  No disponible")

    print("\n⚙️ Response format:")
    print(f"  {getattr(tool, 'response_format', 'N/A')}")

    print("\n🔧 Tipo:")
    print(f"  {type(tool)}")

    print("-" * 50)


async def main():

    client = MultiServerMCPClient(
        ## TODO
    )

    # Nos descargamos las herramientas
    tools = await client.get_tools()

    try:
        prompts = await client.get_prompt("weather", "nombrePrompt1") # Ejemplo
    except Exception as e:
        print("No existe el prompt con el nombre asociado")
    
    try:
        resources = await client.get_resources()
    except Exception as e:
        print("No existen recursos en este MCP")

    for tool in tools:
       pretty_tool(tool)
    
    agente = create_agent(
        model=ChatOllama(model="gemma4:e2b", reasoning=True), # Usad alguno vuestro
        tools=tools,
        system_prompt="Eres un asistente que llama a herramientas. " \
        "Devuelve la salida en español si la tool devuelve la información en inglés."
    )


    while (prompt := input("> ")) != "end":
       async for paso in agente.astream({
            "messages": [
                HumanMessage(prompt)
            ]
        }, stream_mode="values"):
            ultimo_mensaje = paso["messages"][-1]

            hayRazonamiento = ""
            if hasattr(ultimo_mensaje, "additional_kwargs"): # sí, asi de escondido está el razonamiento
                hayRazonamiento = ultimo_mensaje.additional_kwargs.get("reasoning_content", "")

            if hayRazonamiento:
                print("\n=== PENSANDO ===")
                print(hayRazonamiento)

            print("\n=== MENSAJE ===")
            ultimo_mensaje.pretty_print()


# Lanzamos de forma concurrente. Es como la clase Thread de Java
asyncio.run(main())