import unittest
from unittest.mock import patch, MagicMock

# Import the function to test
from whatsapp_bot.helper.openai_api import chat_completion

class TestOpenAIApi(unittest.TestCase):

    @patch('whatsapp_bot.helper.openai_api.client')  # Mock the OpenAI client
    def test_chat_completion_success(self, mock_client):
        """
        Test chat_completion returns a valid response when OpenAI client succeeds.
        """
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_response

        # Call the function
        result = chat_completion("Hello!")

        # Validate the result
        self.assertEqual(result, "Test response")

        # Ensure OpenAI client was called with correct parameters
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}]
        )

    @patch('whatsapp_bot.helper.openai_api.client')  # Mock the OpenAI client
    def test_chat_completion_failure(self, mock_client):
        """
        Test chat_completion returns the error message when OpenAI client fails.
        """
        # Mock OpenAI to raise an exception
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        # Call the function
        result = chat_completion("Hello!")

        # Validate the result is the ERROR_MESSAGE
        from project_config.config import ERROR_MESSAGE
        self.assertEqual(result, ERROR_MESSAGE)

        # Ensure OpenAI client was called with correct parameters
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}]
        )


if __name__ == '__main__':
    unittest.main()
