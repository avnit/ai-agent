import os
import sys
import vertexai
from vertexai import agent_engines as ae
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the agent from the modelarmor module
from modelarmor.agent import root_agent

# Initialize the Vertex AI SDK
vertexai.init(
    project="YOUR_PROJECT_ID",  # Replace with your Google Cloud project ID
    location="YOUR_REGION",     # Replace with your desired region (e.g., "us-central1")
)

# Build and deploy the agent
agent = ae.AdkApp(
    agent=root_agent,
    enable_tracking=True,
)

print(f"Agent '{agent.name}' deployed with resource name: '{agent.resource_name}'")