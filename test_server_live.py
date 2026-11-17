"""
Tests live Gradio server startup, HTTP connectivity, and page response.
"""

import time
import urllib.request
import urllib.error
from app import create_dashboard, CUSTOM_CSS
import gradio as gr

def test_live_server():
    print("Initializing Gradio app for live HTTP test...")
    demo = create_dashboard()

    print("Launching demo on http://127.0.0.1:7865 (prevent_thread_lock=True)...")
    server_name = "127.0.0.1"
    server_port = 7865

    # Launch without blocking thread
    app, local_url, share_url = demo.launch(
        server_name=server_name,
        server_port=server_port,
        prevent_thread_lock=True,
        share=False,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS
    )

    print(f"Server launched at: {local_url}")
    time.sleep(2)

    try:
        req = urllib.request.Request(f"http://{server_name}:{server_port}/")
        with urllib.request.urlopen(req, timeout=5) as response:
            status_code = response.getcode()
            content = response.read().decode('utf-8')
            print(f"HTTP GET returned status code: {status_code}")
            assert status_code == 200, f"Expected status 200, got {status_code}"
            assert "AI People Counting & Flow Analytics" in content or "Gradio" in content, "Dashboard title missing from HTML"
            print("Successfully received HTML page from local Gradio server!")
    except Exception as e:
        print(f"FAILED to probe server: {e}")
        demo.close()
        raise e

    print("Closing test server...")
    demo.close()
    print("Server closed cleanly. HTTP Live Test PASSED!")
    return True

if __name__ == "__main__":
    test_live_server()

