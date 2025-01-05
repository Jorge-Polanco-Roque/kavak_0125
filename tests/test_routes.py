import unittest
from unittest.mock import patch, MagicMock
from whatsapp_bot.src.main import app

class TestRoutes(unittest.TestCase):

    def setUp(self):
        """
        Set up the test client for the Flask application.
        """
        self.app = app.test_client()
        self.app.testing = True

    def test_home_route(self):
        """
        Test the home route '/' returns 'OK'.
        """
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode('utf-8'), 'OK')

    @patch('whatsapp_bot.src.main.chat_completion')  # Mock chat_completion
    @patch('whatsapp_bot.src.main.send_message')  # Mock send_message
    def test_twilio_route(self, mock_send_message, mock_chat_completion):
        """
        Test the Twilio webhook route '/twilio' processes requests correctly.
        """
        # Mock chat_completion response
        mock_chat_completion.return_value = "Mocked response"

        # Simulate incoming Twilio POST request
        data = {
            "From": "whatsapp:+5218119759893",
            "Body": "Hello!"
        }
        response = self.app.post('/twilio', data=data)

        # Assert the response is OK
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode('utf-8'), 'OK')

        # Validate chat_completion was called with the correct query
        mock_chat_completion.assert_called_once_with("Hello!")

        # Validate send_message was called with the correct arguments
        mock_send_message.assert_called_once_with(
            "whatsapp:+5218119759893", "Mocked response"
        )

    @patch('whatsapp_bot.src.main.chat_completion')  # Mock chat_completion
    def test_twilio_route_missing_body(self, mock_chat_completion):
        """
        Test the Twilio webhook route '/twilio' handles missing 'Body' gracefully.
        """
        # Simulate incoming Twilio POST request without 'Body'
        data = {
            "From": "whatsapp:+5218119759893"
        }
        response = self.app.post('/twilio', data=data)

        # Assert the response is OK
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode('utf-8'), 'OK')

        # Validate chat_completion was not called
        mock_chat_completion.assert_not_called()

if __name__ == '__main__':
    unittest.main()
