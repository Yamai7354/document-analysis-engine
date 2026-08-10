"""
Shared chat-LLM factory. Reads llm.provider/model/temperature from
config.yaml and OPENAI_API_KEY / ANTHROPIC_API_KEY from the environment
(loaded from .env if present) so retrieval.py and planner.py don't each
duplicate provider-selection logic.
"""
from dotenv import load_dotenv

# override=True: this project's .env should win over any ambient shell
# environment variables of the same name (e.g. a stray OPENROUTER_API_KEY
# left over from something else on the machine).
load_dotenv(override=True)

from config import CONFIG


def get_chat_llm(tools=None):
    """
    Returns a configured chat model for the provider set in config.yaml
    (llm.provider: "openai" | "anthropic"), optionally bound to `tools`
    for function calling.
    """
    cfg = CONFIG.get("llm", {})
    provider = cfg.get("provider", "openai")
    model = cfg.get("model")
    temperature = cfg.get("temperature", 0.2)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=model, temperature=temperature)
    elif provider == "openrouter":
        import os
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model, temperature=temperature)

    if tools:
        llm = llm.bind_tools(tools)
    return llm
