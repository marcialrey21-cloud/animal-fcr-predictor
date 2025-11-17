from web_app import create_app

# Gunicorn calls this to start the application.
app = create_app()