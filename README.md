# Introduction
A collection of methods and approaches to build and serve chatbots and agents

# Requirement
- To work with Databricks, you need Premium / Paid Databricks workspace(s) to enable interating with the workspace via REST API with token (SP or PAT). Developer access is not enabled for Databricks Free Edition.
    - Generate a PAT: Databricks workspace > User icon > settings > Developer > Access tokens > Generate new token > Set expiry date, name, scope (follow least-privilege) > Save the token to a safe place.
    - Recommended starting scopes: `mlflow`, `model-serving`
    - For more info on scopes, refer to: https://docs.databricks.com/api/workspace/api/scopes
- Databricks Free Edition can allow workspace access via the `mlflow[databricks]` python SDK with embedded profile (U2M) in the active environment.