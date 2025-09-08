#%%
import os
from dotenv import load_dotenv
from typing import Optional
from google.genai import types
from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.models import LlmResponse, LlmRequest
from google.adk.agents.callback_context import CallbackContext
from google.cloud import aiplatform_v1beta1 as aiplatform
from google.protobuf import struct_pb2

load_dotenv()

project = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION")
endpoint_id = os.getenv("AIP_ENDPOINT_ID")

client = aiplatform.PredictionServiceClient(client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"})


def model_armor_analyze(prompt: str):
    instance = struct_pb2.Struct()
    instance.fields["prompt"].string_value = prompt
    instances = [instance]
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances
    )
    print(response)
    
    jailbreak = None
    sensitive_data = None

    for prediction in response.predictions:
        if "jailbreak" in prediction:
            jailbreak = prediction
        if "sensitive_data" in prediction:
            sensitive_data = prediction

    return jailbreak, sensitive_data


def guardrail_function(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    agent_name = callback_context.agent_name
    print(f"[Callback] Before model call for agent: {agent_name}")

    pii_found = callback_context.state.get("PII", False)

    last_user_message = ""
    if llm_request.contents and llm_request.contents[-1].role == 'user':
        if llm_request.contents[-1].parts:
            last_user_message = llm_request.contents[-1].parts[0].text
    print(f"[Callback] Inspecting last user message: '{last_user_message}'")

    last_user_message_lower = last_user_message.lower()

    if pii_found:
        if last_user_message_lower == "yes":
            callback_context.state["PII"] = False
            return None
        else:
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="Please respond Yes/No to continue")]
                )
            )

    jailbreak, sensitive_data = model_armor_analyze(last_user_message)
    if sensitive_data and sensitive_data["sensitive_data"]["match"]:
        callback_context.state["PII"] = True
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=
                                  f"""
                                  Your query has identify the following personal information:
                                  {sensitive_data["sensitive_data"]["info_types"]}
                                  
                                  Would you like to continue? (Yes/No)
                                  """
                                  )],
            )
        )

    if jailbreak and jailbreak["jailbreak"]["match"]:
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="""Break Reason: Jailbreak""")]
            )
        )
    return None


root_agent = Agent(
    name="root_agent",
    model="gemini-2.5-flash",
    description="You are an Artificial General Intelligence",
    instruction="Answer any question using your `google_search_tool` as your grounding",
    before_model_callback=guardrail_function,
    tools=[google_search]
)