from state import SummaryState
from agent import run_agent

if __name__ == "__main__":
    text = """
    - AI agents must follow deterministic workflows
    - Tools allow structured actions
    - State enforces order
    """

    state = SummaryState(text=text)

    while state.step != "done":
        state = run_agent(state)

    print("\n\n=== FINAL SUMMARY ===")
    print(state.final_summary)