#!/usr/bin/env python
"""Smart launcher for HDP PREDICTOR that finds an available port"""
import socket
import subprocess
import sys

def find_available_port(start_port=8501):
    """Find an available port starting from start_port"""
    port = start_port
    max_attempts = 50
    
    for _ in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                s.listen(1)
                print(f"✅ Port {port} is available")
                return port
        except OSError:
            port += 1
    
    raise RuntimeError(f"Could not find an available port between {start_port} and {start_port + max_attempts}")

if __name__ == '__main__':
    port = find_available_port()
    print(f"🚀 Starting HDP PREDICTOR on port {port}...")
    print(f"📱 Access the app at: http://localhost:{port}")
    print("-" * 60)
    
    # Run streamlit with the found port
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run', 'app.py',
        '--server.port', str(port),
        '--server.headless', 'true'
    ])
