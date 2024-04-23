import requests

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
# output = query({
# 	"inputs": "Can you please let us know more details about your ",
# })

class HuggingFaceServerlessComponent(CustomComponent):
    display_name: str = "Hugging Face Inference API"
    description: str = "LLM model from Hugging Face Inference API."

    def build_config(self):
        return {
            "payload": {"display_name": "Payload"},
            "API_URL": {"display_name": "Endpoint URL", "password": True},
            "huggingfacehub_api_token": {"display_name": "API token", "password": True},
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
