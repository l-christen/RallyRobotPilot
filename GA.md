# How to Launch the GA Pipeline

This process requires two separate terminals running on the server.

---

## Terminal 1: Run the Server (Docker)

This terminal runs the headless simulation and the API.

1.  In your project's root directory (`~/RallyRobotPilot`), build and run the Docker container:

    ```bash
    # Use this command to build AND run
    docker-compose up --build headless
    ```
    *Alternatively, if you know no files changed, just run:*
    ```bash
    docker-compose up headless
    ```

2.  **Wait** for the server to initialize. The terminal will show a lot of output.

3.  ✅ **Do not proceed** until you see the log messages confirming the server is ready:
    * `Preparing game application...`
    * `Starting Flask server on 0.0.0.0:5000...`
    * `Waiting for connections`

4.  After finishing your GA run, you can stop the server with `CTRL + C` in this terminal. Then use :
    ```bash
    docker-compose down
    ```
    to clean up the Docker resources.

---

## Terminal 2: Run the Client (GA Script)

This terminal runs your `hello_world` script, which connects to the server in Terminal 1.

1.  Open a **second terminal** on the same machine.

2.  Navigate to the directory containing your script:
    ```bash
    cd ~/RallyRobotPilot/scripts
    ```

3.  Run the client script:
    ```bash
    python3 run_hello_world.py
    ```

4.  ✅ **Observe the output.** You should see the client connect and run the test:
    * `--- Starting 'Hello World' GA Client ---`
    * `Sending 'reset' command...`
    * (A series of steps...)
    * `--- FINAL FITNESS REPORT ---`