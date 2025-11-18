# This script is for local play to record a human-driven run.
# It does NOT start the Flask server or RemoteController.

from rallyrobopilot.game_launcher import prepare_game_app

# 1. Prepare the Ursina game app and car
print("Preparing game application for local recording...")

# This loads the game, track, and car.
# It will use your modified car.py (for recording)
# and your modified track.py (for visible checkpoints).
app, car = prepare_game_app()
print("Game application prepared.")

# 2. Run the Ursina game
print("Starting Ursina game...")
print("!!! DRIVE ONE FULL LAP TO SAVE YOUR 'human_seed.json' FILE !!!")
print("!!! Your keyboard will work now. !!!")
app.run()