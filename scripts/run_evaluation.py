import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.agents.langgraph_runner import run_langgraph_question

INPUT_CSV = PROJECT_ROOT / "data" / "evaluation_questions.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "evaluation_results.csv"


def normalize_expected_tool(value: str):
    return [x.strip() for x in str(value).split("|")]


def main():
    df = pd.read_csv(INPUT_CSV)

    results = []

    for _, row in df.iterrows():
        question = row["question"]
        expected_tool = str(row["expected_tool"])

        try:
            final_answer, selected_tools, tool_results, resolved_question = run_langgraph_question(question)

            expected_tools = normalize_expected_tool(expected_tool)
            selected_tools_set = set(selected_tools)
            expected_tools_set = set(expected_tools)

            match = expected_tools_set.issubset(selected_tools_set)

            results.append({
                "question": question,
                "expected_tool": expected_tool,
                "selected_tools": "|".join(selected_tools),
                "tool_match": "YES" if match else "NO",
                "resolved_question": resolved_question,
                "final_answer": final_answer[:500]
            })

        except Exception as e:
            results.append({
                "question": question,
                "expected_tool": expected_tool,
                "selected_tools": "ERROR",
                "tool_match": "NO",
                "resolved_question": "",
                "final_answer": f"ERROR: {e}"
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)

    total = len(out_df)
    correct = (out_df["tool_match"] == "YES").sum()
    accuracy = (correct / total) * 100 if total else 0

    print(f"Evaluation completed.")
    print(f"Total questions: {total}")
    print(f"Tool-routing matches: {correct}")
    print(f"Routing accuracy: {accuracy:.2f}%")
    print(f"Saved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()