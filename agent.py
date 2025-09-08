# agent.py
from google.adk.agents import Agent

root_agent = Agent(
  name="root_agent",
  model="gemini-2.5-flash"
  description="You are an AI assistant",
  instruction="Answer any question"
)
