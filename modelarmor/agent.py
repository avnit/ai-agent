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
    
    try:
        response = client.sanitize_user_prompt(request=request)
    except Exception as e:
        print(f"Error calling Model Armor: {e}")
        return None, None, None
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
    print(f"[Callback] Guardrail function invoked for agent: {callback_context.agent_name}")
    print(f"[Callback] LLM request: {llm_request}")

    pii_found = callback_context.state.get("PII", False)

    if not llm_request.contents or llm_request.contents[-1].role != 'user' or not llm_request.contents[-1].parts:
        print("[Callback] Invalid request format. Skipping guardrail.")
        return None

    last_user_message_part = llm_request.contents[-1].parts[0]
    if not hasattr(last_user_message_part, 'text'):
        print("[Callback] Last user message part is not text. Skipping guardrail.")
        return None
    last_user_message = last_user_message_part.text
    print(f"[Callback] Inspecting last user message: '{last_user_message}'")

    if pii_found:
        if str(last_user_message).lower().strip() in ["yes", "y"]:
            print("[Callback] PII confirmed by user. Proceeding with the prompt.")
            callback_context.state["PII"] = False
            return None
        elif str(last_user_message).lower().strip() in ["no", "n"]:
            print("[Callback] PII denied by user. Blocking the prompt.")
            callback_context.state["PII"] = False
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="Please rephrase your query without personal information.")],
                )
            )
        else:
            print("[Callback] PII detected. Asking user for confirmation.")
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=
                                      f"""
                                      Your query has identify the following personal information:
                                      {callback_context.state.get("PII_info_types")}
                                      
                                      Would you like to continue? (Yes/No)
                                      """
                                      )],
                )
            )

    jailbreak, sensitive_data,malicious_conntent = model_armor_analyze(str(last_user_message))
    print(f"[Callback] Model Armor analysis results: jailbreak={jailbreak}, sensitive_data={sensitive_data}, malicious_content={malicious_conntent}")
    if sensitive_data and sensitive_data.sdp_filter_result and sensitive_data.sdp_filter_result.inspect_result and sensitive_data.sdp_filter_result.inspect_result.match_state.name == "MATCH_FOUND":
        callback_context.state["PII"] = True
        info_types = list(sensitive_data.sdp_filter_result.deidentify_result.info_types)
        callback_context.state["PII_info_types"] = info_types
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=
                                  f"""
                                  Your query has identify the following personal information:
                                  {info_types}
                                  
                                  Would you like to continue? (Yes/No)
                                  """
                                  )],
            )
        )

    if jailbreak and jailbreak.pi_and_jailbreak_filter_result.match_state.name == "MATCH_FOUND":
        print("[Callback] Jailbreak detected. Blocking the prompt.")
        # if jailbreak.pi_and_jailbreak_filter_result.match_state.name == "MATCH_FOUND":
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="""Break Reason: Jailbreak""")]
            )
        )
    if malicious_conntent and malicious_conntent.malicious_uri_filter_result.match_state.name == "MATCH_FOUND":
        print("[Callback] Malicious content detected. Blocking the prompt.")
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="""Break Reason: Malicious Content""")]
            )
        )
    
    print("[Callback] No issues detected. Proceeding with the prompt.")
    return None


root_agent = Agent(
    name="root_agent",
    model="gemini-2.5-flash",
    description="You are an Artificial General Intelligence",
    instruction="Answer any question using your `google_search_tool` as your grounding",
    before_model_callback=guardrail_function,
    tools=[google_search]
)
