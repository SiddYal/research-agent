"""Opens the app in the browser once the server answers (used by start.bat / start.sh).

Startup takes a while (loading the AI libraries, connecting to the database), so a fixed
delay opens the browser too early and it shows "can't reach this page".
"""

import sys
import time
import urllib.request
import webbrowser

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
deadline = time.monotonic() + 180
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=2):
            webbrowser.open(url)
            break
    except OSError:
        time.sleep(0.5)
else:
    print(f"\nThe server didn't answer within 3 minutes. Check the messages above, then open {url}.")
