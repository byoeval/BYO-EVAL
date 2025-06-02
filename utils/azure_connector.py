import os
from openai import AzureOpenAI



def setup_environment(model_name="GPT41m"):
    """
    Set up environment and initialize the AzureOpenAI client.

    Parameters:
    - model_name (string): Flag indicating the model to use.

    Returns:
    - client: Initialized AzureOpenAI client.
    """

    deployment_name_key = f"GPT_DEPLOYMENT_NAME_{model_name}"
    os.environ["GPT_DEPLOYMENT_NAME"] = os.getenv(deployment_name_key)
    #print("GPT_DEPLOYMENT_NAME: ", os.environ["GPT_DEPLOYMENT_NAME"])
    client = AzureOpenAI(
            api_key=os.getenv(f"OPENAI_API_KEY_{model_name}"),
            api_version=os.getenv(f"OPENAI_API_VERSION_{model_name}"),
            azure_endpoint=os.getenv(f"OPENAI_API_BASE_{model_name}")
        )

    return client
