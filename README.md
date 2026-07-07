# Introduction
A collection of methods and approaches to build and serve chatbots and agents

# Requirement
## Databricks Remote MLflow Server
- Databricks Free Edition can allow workspace access via the `mlflow[databricks]` python SDK with embedded profile (U2M) in the active environment.
- To use the remote MLflow server in Databricks (including Free Edition) with U2M authentication instead of PAT, create a local profile and use it in the .env file. See a guide here: [Using managed MLflow service for free](https://medium.com/mitb-for-all/using-managed-mlflow-service-for-free-24fab870c75a)
- To work with Databricks with service principals or PAT, you need Premium / Paid Databricks workspace(s).
    - Generate a PAT: Databricks workspace > User icon > settings > Developer > Access tokens > Generate new token > Set expiry date, name, scope (follow least-privilege) > Save the token to a safe place.
    - Recommended starting scopes: `mlflow`, `model-serving`
    - For more info on scopes, refer to: https://docs.databricks.com/api/workspace/api/scopes

## MS Foundry access via Entra ID (OAuth 2)
- Step 1: App registration (one-time, in Azure portal): 
    - Azure Portal → Entra ID → App registrations → New registration. 
    - Note down Tenant ID and Application (Client) ID and use them in arguments in client class init.
    - Then in Certificates & secrets, create a new client secret with expiry date, and note it down
- Step 2: grant RBAC for the app above
    - open the MS Foundry resource of interest
    - Access Control (IAM) → Check access tab / Grant access to this resource / Add role assignment → choose "Cognitive Services User" and click Next → Assign access to / User, group, or service principal / + Select members → search for your app name and click Select → Review and assign.