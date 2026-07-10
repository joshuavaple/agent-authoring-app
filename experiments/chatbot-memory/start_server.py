from mlflow.genai.agent_server import AgentServer
import agent
from agent import LAKEBASE_CONFIG
from utils_memory import (
    lakebase_context,
    set_lakebase_resources,
)
from contextlib import asynccontextmanager
import logging



logger = logging.getLogger(__name__)

# Placeholder for long-running agent server (to deploy in Databricks Apps with HTTP timeout duration)
# agent_server = AgentServer(
#     "ResponsesAgent", 
#     enable_chat_proxy=True,
#     db_instance_name=LAKEBASE_CONFIG.instance_name,
#     db_autoscaling_endpoint=LAKEBASE_CONFIG.autoscaling_endpoint,
#     db_project=LAKEBASE_CONFIG.autoscaling_project,
#     db_branch=LAKEBASE_CONFIG.autoscaling_branch,)

# Using the vanilla server
agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=True)

app = agent_server.app

_original_lifespan = app.router.lifespan_context

@asynccontextmanager
async def _lifespan(app):
    try:
        async with lakebase_context(LAKEBASE_CONFIG) as (checkpointer, store):
            await checkpointer.setup()
            # await store.setup()
            logger.info("Lakebase setup complete")

            app.state.checkpointer = checkpointer
            app.state.store = store

            set_lakebase_resources(checkpointer, store)

            try:
                async with _original_lifespan(app):
                    yield
            except Exception as exc:
                logger.warning(
                    "Long-running DB initialization failed: %s. Background mode disabled.",
                    exc,
                )
                yield
    except Exception as exc:
        error_msg = str(exc).lower()
        if any(
            keyword in error_msg
            for keyword in [
                "lakebase",
                "pg_hba",
                "postgres",
                "database instance",
                "insufficient privilege",
            ]
        ):
            logger.error(
                "Lakebase session setup failed: %s\n\n%s",
                exc,
                # get_lakebase_access_error_message(LAKEBASE_CONFIG.description),
            )
        else:
            logger.error("Lakebase session setup failed: %s", exc, exc_info=True)
        raise


app.router.lifespan_context = _lifespan


def main():

    agent_server.run(app_import_string="start_server:app")


if __name__ == "__main__":
    main()