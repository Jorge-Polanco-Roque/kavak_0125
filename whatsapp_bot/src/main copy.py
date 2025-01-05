from flask import Flask, request, jsonify, send_from_directory
import os
from whatsapp_bot.helper.openai_api import chat_completion
from whatsapp_bot.helper.twilio_api import send_message
from project_config import config

app = Flask(__name__)

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    return 'OK', 200

# Twilio webhook route
@app.route('/twilio', methods=['POST'])
def twilio():
    try:
        # Log received request
        data = request.form.to_dict()
        print(f"Received data: {data}")
        print(f"Headers: {request.headers}")

        # Extract message body and sender ID
        query = data.get('Body')
        sender_id = data.get('From')
        print(f"Sender ID: {sender_id}")
        print(f"Query: {query}")

        # Acknowledge Twilio immediately
        #response_message = "Processing your request..."
        #send_message(sender_id, response_message)

        # Generate response asynchronously
        if query:
            response = chat_completion(query)
            send_message(sender_id, response)

        return jsonify({'status': 'success'}), 200
    except Exception as e:
        print(f"Error in Twilio webhook: {e}")
        return jsonify({'error': str(e)}), 500

# Route to serve robots.txt
@app.route('/robots.txt')
def robots_txt():
    return send_from_directory(os.path.abspath('.'), 'robots.txt')

# Route to serve favicon.ico
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.abspath('.'), 'favicon.ico')

if __name__ == '__main__':
    # Ensure app is accessible for Twilio webhook with debug mode enabled
    app.run(host='0.0.0.0', port=5000, debug=True)
