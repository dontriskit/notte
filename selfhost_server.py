# selfhost_server.py
import asyncio
import os
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from loguru import logger
from notte_agent.falco.agent import FalcoAgent, FalcoAgentConfig
from notte_agent.common.types import AgentResponse
from notte_browser.env import NotteEnvConfig
from notte_core.llms.engine import LlmModel
from notte_sdk.types import AgentRequest # Re-using this for input validation

# Load environment variables from .env file
load_dotenv()

# --- Agent Configuration ---
# Configure the agent to use Gemini and local browser sessions.
# Adapt NotteEnvConfig as needed (e.g., headless mode, web security).
# Ensure cdp_url is None for local execution.
agent_config = (
    FalcoAgentConfig()
    .model(LlmModel.gemini, deep=True)
    .map_env(lambda env: env
             # Correct way to modify the inner window config:
             .set_window(env.window.set_cdp_url(None)) # <-- Use set_window to apply changes to BrowserWindowConfig
             .headless(True)                      # Chain other modifiers as before
             .disable_web_security()              # Chain other modifiers as before
             # .steps(15)                         # Optionally set steps
             )
    .use_vision(False)
)


# Instantiate the agent
# If using HashiCorp Vault, initialize it here and pass it:
# from notte_integrations.credentials.hashicorp.vault import HashiCorpVault
# vault = HashiCorpVault.create_from_env()
# agent = Agent(config=agent_config, vault=vault)
agent = FalcoAgent(config=agent_config)

# --- FastAPI App ---
app = FastAPI(title="Notte Self-Hosted Agent API")

logger.info("Notte Agent Server configured and ready.")
logger.info(f"Agent Config: {agent_config.model_dump_json(indent=2)}")

@app.post("/agent/run", response_model=AgentResponse)
async def run_agent_task(request: AgentRequest):
    """
    Endpoint to run a task with the self-hosted Notte agent.
    """
    logger.info(f"Received task: {request.task} for URL: {request.url}")
    try:
        # The agent internally manages its browser session via NotteEnv
        result = await agent.run(task=request.task, url=request.url)
        logger.info(f"Task completed. Success: {result.success}, Duration: {result.duration_in_s:.2f}s")
        return result
    except Exception as e:
        logger.error(f"Agent run failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI server using uvicorn
    # You might need to adjust host and port depending on your setup
    uvicorn.run(app, host="0.0.0.0", port=8000)