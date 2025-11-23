from openai import OpenAI
from state import SummaryState
from tools import (
    extract_bullets,
    finalize_summary,
    ExtractBulletsInput,
    FinalizeSummaryInput,
)
import os
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def run_agent(state: SummaryState) -> SummaryState:

    # ===== STEP 1: LLM decides what to do next =====
    if state.step == "plan":
        print("\n[AGENT] Step 1: LLM Planning...")

        plan_prompt = f"""
        You are a deterministic workflow agent.
        Current state:

        {state.model_dump_json(indent=2)}

        RULES:
        - If bullets == null → next_step MUST be "extract"
        - If bullets is a list AND final_summary == null → next_step MUST be "summarize"
        - If final_summary != null → next_step MUST be "done"

        Respond ONLY with strict JSON:
        {{"next_step": "<extract|summarize|done>"}}
        """

        resp = client.responses.create(
            model="gpt-5-nano",
            input=plan_prompt
        )

        # try to parse JSON
        try:
            next_step = json.loads(resp.output_text)["next_step"]
        except:
            print("[ERROR] invalid JSON — defaulting to extract")
            next_step = "extract"
        print(f'current state: {state}')
        print(f'agent determined next_step: {next_step}')
        print(f'bullets: {state.bullets}')
        # HARD VALIDATION (enforce determinism)
        if state.bullets is None:
            state.step = "extract"
        elif state.bullets is not None and state.final_summary is None:
            state.step = "summarize"
        elif state.final_summary is not None:
            state.step = "done"
        else:
            print("[ERROR] unexpected state — defaulting to extract")
            state.step = "extract"

        return state


    # ===== STEP 2: EXTRACT BULLETS =====
    if state.step == "extract":
        print("\n[AGENT] Step 2: LLM choosing TOOL + executing it...")

        tool_prompt = f"""
        You are a deterministic workflow agent.
        The current state is:

        {state.model_dump_json(indent=2)}

        You MUST choose exactly one tool to call:
        - extract_bullets

        Respond ONLY in this JSON format:
        {{
        "tool": "<tool name>",
        "arguments": {{ ... }}
        }}
        """

        resp = client.responses.create(
            model="gpt-5-nano",
            input=tool_prompt
        )
        print("\n")
        print(f'agent response for step two: {resp.output_text}')
        print("\n")
        # parse LLM output
        try:
            parsed = json.loads(resp.output_text)
            print(parsed)
            tool_name = parsed["tool"]
            args = parsed["arguments"]
        except:
            print("[ERROR] invalid JSON — retrying with direct python call")
            tool_name = "extract_bullets"
            args = {"text": state.text}

        # route call
        if tool_name == "extract_bullets":
            tool_input = ExtractBulletsInput(**args)
            tool_output = extract_bullets(tool_input)
            state.bullets = tool_output.bullets
        print(f'tool_input: {tool_input}')
        print(f'tool output: {tool_output}')
        print((f'output from agent call will be in parsed["arguments"] = {args}'))
        state.step = "summarize"
        return state


    # ===== STEP 3: SUMMARIZE =====
    if state.step == "summarize":
        print("\n[AGENT] Step 3: LLM choosing TOOL + executing it...")

        tool_prompt = f"""
        You are at the summarize step.
        The current state is:

        {state.model_dump_json(indent=2)}

        Choose exactly one tool to call:
        - finalize_summary

        Respond ONLY in JSON:
        {{
        "tool": "<tool name>",
        "arguments": {{ ... }}
        }}
        """

        resp = client.responses.create(
            model="gpt-5-nano",
            input=tool_prompt
        )
        print(resp.output_text)

        try:
            parsed = json.loads(resp.output_text)
            tool_name = parsed["tool"]
            args = parsed["arguments"]
        except:
            print("[ERROR] invalid JSON — retrying with direct python call")
            tool_name = "finalize_summary"
            args = {"bullets": state.bullets}

        if tool_name == "finalize_summary":
            tool_input = FinalizeSummaryInput(**args)
            tool_output = finalize_summary(tool_input)
            state.final_summary = tool_output.summary
        print(f'tool_input: {tool_input}')
        print(f'tool output: {tool_output}')
        print((f'output from agent call will be in parsed["arguments"] = {args}'))
        state.step = "done"
        return state


    return state