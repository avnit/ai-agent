#%%
import os
from dotenv import load_dotenv
from typing import Optional
from google.genai import types
from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.models import LlmResponse, LlmRequest
from google.adk.agents.callback_context import CallbackContext
from google.api_core.client_options import ClientOptions
from google.cloud import modelarmor_v1 as aiplatform
from google.protobuf import struct_pb2

load_dotenv()

project = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION")
endpoint_id = os.getenv("AIP_ENDPOINT_ID")

client = aiplatform.ModelArmorClient(
    transport="rest",
    client_options=ClientOptions(api_endpoint=f"modelarmor.{location}.rep.googleapis.com")
)


def model_armor_analyze(prompt: str):
    print(f"Analyzing prompt with Model Armor: {prompt}")
    print(f"Using Model Armor endpoint: projects/{project}/locations/{location}/templates/{endpoint_id}")
    user_prompt_data = aiplatform.DataItem(text=prompt)
    # user_prompt_data.text = prompt
    request = aiplatform.SanitizeUserPromptRequest(
        name = f"projects/{project}/locations/{location}/templates/{endpoint_id}",
        user_prompt_data=user_prompt_data    
    )
    
    response = client.sanitize_user_prompt(request=request)
    print(response)
    
    jailbreak = response.sanitization_result.filter_results.get("pi_and_jailbreak")
    sensitive_data = response.sanitization_result.filter_results.get("sdp")
    malicious_content = response.sanitization_result.filter_results.get("malicious_uris")
    # for prediction in response.predictions:
    #     if "jailbreak" in prediction:
    #         jailbreak = prediction
    #     if "sensitive_data" in prediction:
    #         sensitive_data = prediction

    return jailbreak, sensitive_data, malicious_content


def guardrail_function(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    agent_name = callback_context.agent_name
    print(f"[Callback] Before model call for agent: {agent_name}")

    pii_found = callback_context.state.get("PII", False)

    last_user_message = ""
    if llm_request.contents and llm_request.contents[-1].role == 'user':
        if llm_request.contents[-1].parts:
            last_user_message = llm_request.contents[-1].parts[0].text
    print(f"[Callback] Inspecting last user message: '{last_user_message}'")

    if pii_found and str(last_user_message).lower() != "yes":
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="Please respond Yes/No to continue")]
            )
        )
    elif pii_found and str(last_user_message).lower() == "yes":
        callback_context.state["PII"] = False
        return None

    jailbreak, sensitive_data,malicious_conntent = model_armor_analyze(str(last_user_message))
    if sensitive_data and sensitive_data.sdp_filter_result and sensitive_data.sdp_filter_result.inspect_result:
        if sensitive_data.sdp_filter_result.inspect_result.match_state.name == "MATCH_FOUND":
            pii_found = True
            callback_context.state["PII"] = True
            if pii_found and str(last_user_message).lower() != "No":
                return LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text=
                                          f"""
                                          Your query has identify the following personal information:
                                          {sensitive_data.sdp_filter_result.deidentify_result.info_types}
                                          
                                          Would you like to continue? (Yes/No)
                                          """
                                          )],
                    )
                )
            elif pii_found and str(last_user_message).lower() == "Yes":
                callback_context.state["PII"] = False
                return None
            elif pii_found and str(last_user_message).lower() == "No":
                callback_context.state["PII"] = False
                return LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text="Please rephrase your query without personal information.")],
                    )
                )

    if jailbreak and jailbreak.pi_and_jailbreak_filter_result.match_state.name == "MATCH_FOUND":
        # if jailbreak.pi_and_jailbreak_filter_result.match_state.name == "MATCH_FOUND":
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="""Break Reason: Jailbreak""")]
            )
        )
    if malicious_conntent and malicious_conntent.malicious_uri_filter_result.match_state.name == "MATCH_FOUND":
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="""Break Reason: Malicious Content""")]
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
