import requests
from typing import Optional
from langchain.llms.base import BaseLLM
from langflow import CustomComponent

def query(payload, api_url, huggingfacehub_api_token):
	response = requests.post(api_url, headers='"Authorization": "Bearer {huggingfacehub_api_token}"', json=payload)
	return response.json()
	
# output = query({
# 	"inputs": "Can you please let us know more details about your ",
# })

class HuggingFaceEndpointsComponent(CustomComponent):
    display_name: str = "Hugging Face Inference API"
    description: str = "LLM model from Hugging Face Inference API."

    def build_config(self):
        return {
            payload: {
              display_name: "Payload",
              required: True,
            },
            API_URL: {
              "display_name": "Endpoint URL",
              "required": True,
            },
            "huggingfacehub_api_token": {
              "display_name": "API token",
              "password": True,
              "required": True,
            },
        }

    def build(
        self,
        payload,
        api_url: str,
        huggingfacehub_api_token: Optional[str] = None,
    ) -> BaseLLM:
        try:
            output = query(  # type: ignore
                payload=payload,
                api_url=api_url,
                huggingfacehub_api_token:huggingfacehub_api_token,
            )
        except Exception as e:
            raise ValueError("Could not connect to HuggingFace Endpoints API.") from e
        return output
