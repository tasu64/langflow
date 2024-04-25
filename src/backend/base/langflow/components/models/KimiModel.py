import requests
from typing import Optional
from langflow.field_typing import Text
from langflow import CustomComponent

class KimiComponent(CustomComponent):
  display_name: str = "Kimi"
  description: str = "LLM model from Kimi."
  api_url = "https://api.moonshot.cn/v1/chat/completions"
  model_name = "moonshot-v1-8k"
  history = []

  def build_config(self):
    return {
      "payload": {
        "display_name": "Payload",
        "required": True,
      },
      "kimi_api_token": {
        "display_name": "API token",
        "password": True,
        "required": True,
      },
      "temperature": {
        "display_name": "Temperature",
        "value": 0.1,
      },
    }

  def query(self, payload, kimi_api_token, temperature):
    headers = {
      "Authorization": f"Bearer {kimi_api_token}"
    }
    data = {
      "model": self.model_name,
      "message": [{
        "role": "user",
        "content": payload,
      }],
      "temperature": temperature,
    }
    response = requests.post(self.api_url, headers=headers, json=data)
    return response.json()

  def build(
    self,
    payload: Text,
    kimi_api_token: Optional[str] = None,
    temperature: float = 0.1,
  ) -> Text:
    try:
      output = self.query(payload=payload, kimi_api_token=kimi_api_token, temperature=temperature)
    except Exception as e:
      raise ValueError("Could not connect to Kimi Serveless API.") from e
    return output
