from app.agents.langgraph_flow import build_langgraph

graph = build_langgraph()

questions = [
    "Which items are below reorder threshold?",
    "Which shipments are delayed?",
    "What is the process for damaged goods?",
    "Which items should be restocked first?"
]

for question in questions:
    print("\n" + "=" * 80)
    print("QUESTION:", question)
    result = graph.invoke({"user_question": question})
    print("SELECTED TOOLS:", result.get("selected_tools"))
    print("FINAL ANSWER:")
    print(result.get("final_answer"))