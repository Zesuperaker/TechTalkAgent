import os
from flask import Flask
from routes import register_routes
from vector_db import init_chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# Initialize Chroma on app startup
init_chroma()

# Register all routes
register_routes(app)

if __name__ == '__main__':
    app.run(debug=True, port=5000)