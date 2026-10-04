import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


class GeminiLLM:
    """
    Wrapper around the Gemini language model.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.8-flash",
        temperature: float = 0.2,
    ):
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. "
                "Please add it to the .env file."
            )

        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
        )

    def generate(self, prompt: str) -> str:
      """
      Generate a response from Gemini.
      """

      response = self.model.invoke(prompt)

      if isinstance(response.content, str):
          return response.content

      if isinstance(response.content, list):
          text_parts = []

          for block in response.content:
              if isinstance(block, dict) and block.get("type") == "text":
                  text_parts.append(block.get("text", ""))

          return "\n".join(text_parts)

      return str(response.content)