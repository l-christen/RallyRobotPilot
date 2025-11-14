from rallyrobopilot import prepare_game_app

app, car = prepare_game_app()
car._record_enabled = False
app.run()
