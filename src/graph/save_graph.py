from langgraph.graph.state import CompiledStateGraph


def save_graph(graph: CompiledStateGraph, indx=None):
    app = graph.get_graph().draw_mermaid_png()
    with open(fr"D:\codes\GenAIPoCs\research-agent\lg/grap_{indx}.png", 'wb') as f:
        f.write(app)

    print(f"Graph saved :: graph_{indx}.png")
