from flask import render_template, request, jsonify, send_file, Response
from services import chat_with_agent, chat_with_agent_stream, process_uploaded_text, get_kb_status, clear_conversation, get_audio_response
from io import BytesIO
import struct
import json


def wrap_pcm_as_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
    """Wrap raw PCM16 audio data with WAV header so it can be played in browsers"""
    if not pcm_data:
        return None

    bytes_per_sample = 2  # 16-bit = 2 bytes
    byte_rate = sample_rate * channels * bytes_per_sample
    block_align = channels * bytes_per_sample
    subchunk2_size = len(pcm_data)
    chunk_size = 36 + subchunk2_size

    # Build WAV header
    wav_header = b'RIFF'
    wav_header += struct.pack('<I', chunk_size)
    wav_header += b'WAVE'
    wav_header += b'fmt '
    wav_header += struct.pack('<I', 16)  # Subchunk1Size
    wav_header += struct.pack('<H', 1)  # AudioFormat (1 = PCM)
    wav_header += struct.pack('<H', channels)
    wav_header += struct.pack('<I', sample_rate)
    wav_header += struct.pack('<I', byte_rate)
    wav_header += struct.pack('<H', block_align)
    wav_header += struct.pack('<H', 16)  # BitsPerSample
    wav_header += b'data'
    wav_header += struct.pack('<I', subchunk2_size)

    return wav_header + pcm_data


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
        """API endpoint for chat messages with streaming"""
        data = request.json
        user_message = data.get('message', '')

        if not user_message.strip():
            return jsonify({'error': 'Empty message'}), 400

        def generate():
            """Generator function for streaming response"""
            try:
                for token in chat_with_agent_stream(user_message):
                    yield f"data: {json.dumps({'token': token})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

    @app.route('/api/chat/audio', methods=['POST'])
    def api_chat_audio():
        """API endpoint to get audio version of text response"""
        data = request.json
        text = data.get('text', '')

        if not text.strip():
            return jsonify({'error': 'Empty text'}), 400

        try:
            audio_bytes = get_audio_response(text)

            if not audio_bytes:
                return jsonify({'error': 'Failed to generate audio'}), 500

            # Wrap PCM16 in WAV container so browser can play it
            wav_data = wrap_pcm_as_wav(audio_bytes)

            return send_file(
                BytesIO(wav_data),
                mimetype='audio/wav',
                as_attachment=False
            )
        except Exception as e:
            return jsonify({'error': str(e)}), 400

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