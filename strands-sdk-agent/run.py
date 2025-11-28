import asyncio
import logging
from strands_agents import run_agent

from agent import SummaryAgent
from state import SummaryState


async def main():

    initial_state = SummaryState()

    agent = SummaryAgent(state=initial_state)

    input_text = """
    - AI agents must follow deterministic workflows
    - Tools allow structured actions
    - State enforces order
    """

    result = await run_agent(agent, input_text)

    print("\n=== FINAL SUMMARY ===")
    print(result.state.final_summary)


if __name__ == "__main__":
    asyncio.run(main())
