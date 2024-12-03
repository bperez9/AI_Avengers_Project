# tools/base_tool.py

import json
from abc import ABC, abstractmethod
from typing import Dict, List

class BaseTool(ABC):
    """
    Base class for defining custom tools (functions) for use with the LLM.
    """

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the tool."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return the description of the tool."""
        pass

    @abstractmethod
    def get_params_definition(self) -> Dict[str, dict]:
        """Return the parameters definition of the tool."""
        pass

    @abstractmethod
    def run(self, messages: List[dict]) -> List[dict]:
        """Run the tool."""
        pass

class SingleMessageTool(BaseTool):
    """
    Helper class to handle tools that take a single message.
    """

    def get_function_definition(self) -> dict:
        return {
            "name": self.get_name(),
            "description": self.get_description(),
            "parameters": {
                "type": "object",
                "properties": self.get_params_definition(),
                "required": [
                    name for name, param in self.get_params_definition().items() if param.get("required")
                ],
            },
        }

    def run(self, messages: List[dict]) -> List[dict]:
        assert len(messages) == 1, "Expected single message"

        message = messages[0]
        tool_call = message['tool_calls'][0]

        try:
            arguments = json.loads(tool_call['arguments'])
            response = self.run_impl(**arguments)
            response_str = json.dumps(response, ensure_ascii=False)
        except Exception as e:
            response_str = f"Error when running tool: {e}"

        tool_response_message = {
            "tool_call_id": tool_call['id'],
            "role": "tool",
            "name": tool_call['function']['name'],
            "content": response_str,
        }

        return [tool_response_message]

    @abstractmethod
    def run_impl(self, *args, **kwargs):
        """Implement the tool logic."""
        pass