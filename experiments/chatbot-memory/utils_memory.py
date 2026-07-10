import os
import logging
from databricks_langchain import AsyncCheckpointSaver, AsyncDatabricksStore
from typing import Optional, Tuple
from dataclasses import dataclass
from databricks.sdk import WorkspaceClient
from contextlib import asynccontextmanager
from mlflow.types.responses import ResponsesAgentRequest


logger = logging.getLogger(__name__)

# Long-lived Lakebase resources, opened once at app startup in start_server.py's
# lifespan and reused across all requests.
_lakebase_resources: Optional[Tuple[AsyncCheckpointSaver, AsyncDatabricksStore]] = None


def set_lakebase_resources(
    checkpointer: AsyncCheckpointSaver, store: AsyncDatabricksStore
) -> None:
    global _lakebase_resources
    _lakebase_resources = (checkpointer, store)

@dataclass(frozen=True)
class LakebaseConfig:
    instance_name: Optional[str]
    autoscaling_endpoint: Optional[str]
    autoscaling_project: Optional[str]
    autoscaling_branch: Optional[str]
    embedding_endpoint: str = "databricks-gte-large-en"  # override via DATABRICKS_EMBEDDING_ENDPOINT
    embedding_dims: int = 1024
    memory_schema: Optional[str] = None

    @property
    def description(self) -> str:
        return self.autoscaling_endpoint or self.instance_name or f"{self.autoscaling_project}/{self.autoscaling_branch}"


def init_lakebase_config() -> LakebaseConfig:
    endpoint = os.getenv("LAKEBASE_AUTOSCALING_ENDPOINT") or None
    raw_name = os.getenv("LAKEBASE_INSTANCE_NAME") or None
    project = os.getenv("LAKEBASE_AUTOSCALING_PROJECT") or None
    branch = os.getenv("LAKEBASE_AUTOSCALING_BRANCH") or None

    has_autoscaling = project and branch
    if not endpoint and not raw_name and not has_autoscaling:
        raise ValueError(
            "Lakebase configuration is required but not set. "
            "Please set one of the following in your environment:\n"
            "  Option 1 (autoscaling endpoint): LAKEBASE_AUTOSCALING_ENDPOINT=<your-endpoint-name>\n"
            "  Option 2 (autoscaling): LAKEBASE_AUTOSCALING_PROJECT=<project> and LAKEBASE_AUTOSCALING_BRANCH=<branch>\n"
            "  Option 3 (provisioned): LAKEBASE_INSTANCE_NAME=<your-instance-name>\n"
        )

    # Priority: endpoint > project+branch > instance_name (mutually exclusive in the library)
    if endpoint:
        instance_name = None
        project = None
        branch = None
    elif has_autoscaling:
        instance_name = None
        endpoint = None
    else:
        instance_name = resolve_lakebase_instance_name(raw_name)
        endpoint = None
        project = None
        branch = None

    embedding_endpoint = os.getenv("DATABRICKS_EMBEDDING_ENDPOINT", "databricks-gte-large-en")
    memory_schema = os.getenv("LAKEBASE_AGENT_MEMORY_SCHEMA") or None
    return LakebaseConfig(
        instance_name=instance_name,
        autoscaling_endpoint=endpoint,
        autoscaling_project=project,
        autoscaling_branch=branch,
        embedding_endpoint=embedding_endpoint,
        memory_schema=memory_schema,
    )


def _is_lakebase_hostname(value: str) -> bool:
    """Check if the value looks like a Lakebase hostname rather than an instance name."""
    # Hostname pattern: instance-{uuid}.database.{env}.cloud.databricks.com
    return ".database." in value and value.endswith(".com")



def resolve_lakebase_instance_name(
    instance_name: str, workspace_client: Optional[WorkspaceClient] = None
) -> str:
    """
    Resolve a Lakebase instance name from a hostname if needed.

    If the input is a hostname (e.g., from Databricks Apps value_from resolution),
    this will resolve it to the actual instance name by listing database instances.

    Args:
        instance_name: Either an instance name or a hostname
        workspace_client: Optional WorkspaceClient to use for resolution

    Returns:
        The resolved instance name

    Raises:
        ValueError: If the hostname cannot be resolved to an instance name
    """
    if not _is_lakebase_hostname(instance_name):
        # Input is already an instance name
        return instance_name

    # Input is a hostname - resolve to instance name
    client = workspace_client or WorkspaceClient()
    hostname = instance_name

    try:
        instances = list(client.database.list_database_instances())
    except Exception as exc:
        raise ValueError(
            f"Unable to list database instances to resolve hostname '{hostname}'. "
            "Ensure you have access to database instances."
        ) from exc

    # Find the instance that matches this hostname
    for instance in instances:
        rw_dns = getattr(instance, "read_write_dns", None)
        ro_dns = getattr(instance, "read_only_dns", None)

        if hostname in (rw_dns, ro_dns):
            resolved_name = getattr(instance, "name", None)
            if not resolved_name:
                raise ValueError(
                    f"Found matching instance for hostname '{hostname}' "
                    "but instance name is not available."
                )
            logging.info(f"Resolved Lakebase hostname '{hostname}' to instance name '{resolved_name}'")
            return resolved_name

    raise ValueError(
        f"Unable to find database instance matching hostname '{hostname}'. "
        "Ensure the hostname is correct and the instance exists."
    )


@asynccontextmanager
async def lakebase_context(config: LakebaseConfig):
    """Yield (checkpointer, store) for short-term and long-term memory.

    Store is not implemented yet; the tuple shape is kept so callers don't
    need to change once AsyncDatabricksStore is wired in.
    """
    async with AsyncCheckpointSaver(
        instance_name=config.instance_name,
        autoscaling_endpoint=config.autoscaling_endpoint,
        project=config.autoscaling_project,
        branch=config.autoscaling_branch,
        schema=config.memory_schema,
    ) as checkpointer:
        # <placeholder for store>
        #     AsyncDatabricksStore(
        #     instance_name=config.instance_name,
        #     autoscaling_endpoint=config.autoscaling_endpoint,
        #     project=config.autoscaling_project,
        #     branch=config.autoscaling_branch,
        #     embedding_endpoint=config.embedding_endpoint,
        #     embedding_dims=config.embedding_dims,
        #     schema=config.memory_schema,
        # ) as store:

        yield checkpointer, None


@asynccontextmanager
async def acquire_lakebase_resources(config: LakebaseConfig):
    """Yield (checkpointer, store) for use in a request handler.

    If start_server.py's lifespan populated the long-lived resources, yield those
    without closing on exit. Otherwise (e.g. evaluate_agent.py running outside the
    FastAPI server) fall back to opening a fresh per-call lakebase_context.
    """
    if _lakebase_resources is not None:
        yield _lakebase_resources
    else:
        async with lakebase_context(config) as resources:
            yield resources
