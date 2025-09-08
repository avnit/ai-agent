# agent.py
from google.adk.agents import Agent

def guardrail_function(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    agent_name = callback_context.agent_name
    print(f"[Callback] Before model call for agent: {agent_name}")

    pii_found = callback_context.state.get("PII", False)

    last_user_message = ""
    if llm_request.contents and llm_request.contents[-1].role == 'user':
        if llm_request.contents[-1].parts:
            last_user_message = llm_request.contents[-1].parts[0].text
    print(f"[Callback] Inspecting last user message: '{last_user_message}'")

    if pii_found and last_user_message.lower() != "yes":
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="Please respond Yes/No to continue")]
            )
        )
    elif pii_found and last_user_message.lower() == "yes":
        callback_context.state["PII"] = False
        return None

    jailbreak, sensitive_data = model_armor_analyze(last_user_message)
    if sensitive_data and sensitive_data.sdp_filter_result and sensitive_data.sdp_filter_result.deidentify_result:
        if sensitive_data.sdp_filter_result.deidentify_result.match_state.name == "MATCH_FOUND":
            pii_found = True
            callback_context.state["PII"] = True
            if pii_found and last_user_message.lower() != "no":
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
            elif pii_found and last_user_message.lower() == "yes":
                callback_context.state["PII"] = False
                return None

    elif jailbreak and jailbreak.pi_and_jailbreak_filter_result:
        if jailbreak.pi_and_jailbreak_filter_result.match_state.name == "MATCH_FOUND":
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