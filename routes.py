from flask import render_template, request, jsonify
from services import chat_with_agent, process_uploaded_text, get_kb_status, clear_conversation


def register_routes(app):
    """Register all Flask routes"""

    @app.route('/')
    def index():
        """Home page - redirect to chat"""
        return render_template('chat.html')

    @app.route('/chat')
    def chat_page():
        """Chat page"""
        return render_template('chat.html')

    @app.route('/upload')
    def upload_page():
        """Upload page"""
        status = get_kb_status()
        return render_template('upload.html', kb_status=status)

    @app.route('/api/chat', methods=['POST'])
    def api_chat():
        """API endpoint for chat messages"""
        data = request.json
        user_message = data.get('message', '')

        if not user_message.strip():
            return jsonify({'error': 'Empty message'}), 400

        response = chat_with_agent(user_message)

        return jsonify({
            'response': response,
            'success': True
        })

    @app.route('/api/chat/clear', methods=['POST'])
    def api_clear_chat():
        """API endpoint to clear conversation history"""
        result = clear_conversation()
        return jsonify(result)

    @app.route('/api/upload', methods=['POST'])
    def api_upload():
        """API endpoint for uploading documents"""
        data = request.json
        text = data.get('text', '')

        if not text.strip():
            return jsonify({'error': 'Empty content'}), 400

        result = process_uploaded_text(text)

        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code

    @app.route('/api/kb-status', methods=['GET'])
    def api_kb_status():
        """API endpoint to get knowledge base status"""
        status = get_kb_status()
        return jsonify(status)

    @app.route('/api/health', methods=['GET'])
    def health():
        """Health check endpoint"""
        return jsonify({'status': 'ok'})