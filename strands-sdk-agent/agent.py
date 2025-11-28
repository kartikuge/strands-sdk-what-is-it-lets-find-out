import os
import logging
from strands_agents import Agent
from strands_agents.events import MessageEvent, ToolResultEvent
from strands_agents_tools.models.openai import OpenAIModelClient

from state import SummaryState
from tools import extract_bullets, finalize_summary

logger = logging.getLogger("strands.agent.SummaryAgent")

class SummaryAgent(Agent):
    state: SummaryState

    def __init__(self, state: SummaryState):
        super().__init__(state)

        logger.debug("Initializing SummaryAgent with empty state")

        self.model = OpenAIModelClient(
            model="gpt-5-nano",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        logger.debug(f"Using model: gpt-5-nano")

    async def handle_message(self, event: MessageEvent):
        logger.info("Received MessageEvent")
        logger.debug(f"Event content: {event.text!r}")

        self.state.text = event.text
        logger.debug(f"Updated state.text: {self.state.text!r}")

        planning = await self.model.complete(f"""
        You are controlling a deterministic workflow agent.
        If text is not empty and bullets is None → respond with "extract".
        If bullets exist and final_summary is None → respond with "summarize".
        Current text:
        {self.state.text}
        Respond ONLY with: extract or summarize.
        """)

        logger.debug(f"Planning LLM output: {planning!r}")

        step = planning.strip().lower()
        logger.info(f"Agent decided next step: {step}")

        if step == "extract":
            logger.info("Calling tool: extract_bullets")
            return await self.call_tool(
                extract_bullets,
                {"text": self.state.text}
            )

        if step == "summarize":
            logger.info("Calling tool: finalize_summary")
            return await self.call_tool(
                finalize_summary,
                {"bullets": self.state.bullets}
            )

    async def handle_tool_result(self, event: ToolResultEvent):
        logger.info(f"ToolResultEvent received from {event.tool_name}")
        logger.debug(f"Tool result payload: {event.result}")

        if event.tool_name == "extract_bullets":
            self.state.bullets = event.result["bullets"]
            logger.debug(f"State updated: bullets={self.state.bullets}")

            logger.info("Calling next tool: finalize_summary")
            return await self.call_tool(
                finalize_summary,
                {"bullets": self.state.bullets}
            )

        elif event.tool_name == "finalize_summary":
            self.state.final_summary = event.result["summary"]
            logger.debug(f"State updated: final_summary={self.state.final_summary!r}")

            logger.info("Workflow complete. Returning final response.")
            return {
                "final_summary": self.state.final_summary
            }
