import unittest
import os
import tempfile
import sys
from unittest.mock import patch

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

class TestConfig(unittest.TestCase):

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test_openai_key',
        'TWILIO_AUTH_TOKEN': 'test_auth_token',
        'TWILIO_SID': 'test_twilio_sid',
        'FROM': 'whatsapp:+14155238886'
    })
    def test_environment_variables_loaded(self):
        import importlib
        from project_config import config
        importlib.reload(config)

        self.assertEqual(config.OPENAI_API_KEY, 'test_openai_key')
        self.assertEqual(config.TWILIO_TOKEN, 'test_auth_token')
        self.assertEqual(config.TWILIO_SID, 'test_twilio_sid')
        self.assertEqual(config.TWILIO_WHATSAPP_FROM, 'whatsapp:+14155238886')

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test_openai_key',
        'TWILIO_AUTH_TOKEN': 'test_auth_token',
        'TWILIO_SID': 'test_twilio_sid',
        'FROM': 'whatsapp:+14155238886'
    })
    def test_output_directory_created(self):
        import importlib
        from project_config import config
        importlib.reload(config)

        self.assertTrue(os.path.exists(config.OUTPUT_DIR))
        self.assertTrue(os.path.isdir(config.OUTPUT_DIR))

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': '',
        'TWILIO_AUTH_TOKEN': 'test_auth_token',
        'TWILIO_SID': 'test_twilio_sid',
        'FROM': 'whatsapp:+14155238886'
    })
    def test_missing_environment_variable(self):
        with self.assertRaises(EnvironmentError) as context:
            import importlib
            from project_config import config
            importlib.reload(config)
        self.assertIn('Missing required environment variable', str(context.exception))


if __name__ == '__main__':
    unittest.main()
