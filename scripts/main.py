import threading
from flask import Flask
from rallyrobopilot import RemoteController
from rallyrobopilot.game_launcher import prepare_game_app

# 1. Create the Flask app
# This is the web server your GA will talk to.
flask_app = Flask(__name__)

# 2. Prepare the Ursina game app and car
# This loads your game, track, and car just as before.
print("Preparing game application...")
app, car = prepare_game_app()
print("Game application prepared.")

# 3. Create the RemoteController
# This is the crucial link. It connects the 'car' object to the 'flask_app'
# and automatically creates the /command and /sensing API routes.
print("Initializing Remote Controller...")
controller = RemoteController(car=car, flask_app=flask_app)
print("Remote Controller initialized.")

# 4. Define a function to run the Flask server
def run_flask():
    """
    Runs the Flask server on host 0.0.0.0 (to be accessible
    outside the Docker container) on port 5000.
    """
    print("Starting Flask server on 0.0.0.0:5000...")
    # We set host='0.0.0.0' so it's reachable from outside Docker
    flask_app.run(host='0.0.0.0', port=5000)

# 5. Run Flask in a separate thread
# This is critical so the API server doesn't block the game.
print("Starting Flask thread...")
# daemon=True ensures this thread automatically exits when the main game quits
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

# 6. Run the Ursina game (this must be on the main thread)
print("Starting Ursina game app.run()...")
app.run()