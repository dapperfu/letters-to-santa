#!/usr/bin/env python3
"""
HTTP server for Letters to Santa website.

This server handles serving static files and receiving video uploads
from the web application. Videos are saved with timestamps to the
videos/ directory.
"""

import os
import sys
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json
import email
from email import message_from_bytes


class LettersToSantaHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Letters to Santa application."""
    
    def do_GET(self: "LettersToSantaHandler") -> None:
        """Handle GET requests for static files."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # Default to index.html for root
        if path == '/':
            path = '/index.html'
        
        # Remove leading slash for file system
        file_path = path.lstrip('/')
        
        # Security: prevent directory traversal
        if '..' in file_path or file_path.startswith('/'):
            self.send_error(403, "Forbidden")
            return
        
        # Check if file exists
        if os.path.exists(file_path) and os.path.isfile(file_path):
            self.send_response(200)
            
            # Set content type based on file extension
            if file_path.endswith('.html'):
                self.send_header('Content-type', 'text/html')
            elif file_path.endswith('.css'):
                self.send_header('Content-type', 'text/css')
            elif file_path.endswith('.js'):
                self.send_header('Content-type', 'application/javascript')
            else:
                self.send_header('Content-type', 'application/octet-stream')
            
            self.end_headers()
            
            # Read and send file
            with open(file_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "File not found")
    
    def do_POST(self: "LettersToSantaHandler") -> None:
        """Handle POST requests for video uploads."""
        if self.path == '/upload':
            try:
                # Get content type
                content_type = self.headers.get('Content-Type', '')
                
                # Get content length
                content_length = int(self.headers['Content-Length'])
                
                # Read the request data
                post_data = self.rfile.read(content_length)
                
                video_data = None
                
                # Handle multipart/form-data
                if 'multipart/form-data' in content_type:
                    # Parse multipart form data
                    msg = message_from_bytes(
                        b'Content-Type: ' + content_type.encode() + b'\r\n\r\n' + post_data
                    )
                    
                    # Extract video data from multipart
                    for part in msg.walk():
                        if part.get_content_type() == 'video/webm':
                            video_data = part.get_payload(decode=True)
                            break
                    
                    if video_data is None:
                        # Try to find any binary part
                        for part in msg.walk():
                            payload = part.get_payload(decode=True)
                            if payload and len(payload) > 1000:  # Likely video data
                                video_data = payload
                                break
                else:
                    # Assume raw video data
                    video_data = post_data
                
                if video_data is None or len(video_data) == 0:
                    raise ValueError("No video data received")
                
                # Generate timestamped filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'video_{timestamp}.webm'
                
                # Ensure videos directory exists
                videos_dir = 'videos'
                os.makedirs(videos_dir, exist_ok=True)
                
                # Save video file
                file_path = os.path.join(videos_dir, filename)
                with open(file_path, 'wb') as f:
                    f.write(video_data)
                
                # Send success response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {
                    'success': True,
                    'filename': filename,
                    'message': 'Video saved successfully'
                }
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                # Send error response
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {
                    'success': False,
                    'error': str(e)
                }
                self.wfile.write(json.dumps(response).encode())
        else:
            self.send_error(404, "Endpoint not found")
    
    def log_message(self: "LettersToSantaHandler", format: str, *args: tuple) -> None:
        """Override to customize log format."""
        sys.stderr.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {format % args}\n")


def run_server(port: int = 8000) -> None:
    """
    Start the HTTP server.
    
    Parameters
    ----------
    port : int
        The port number to run the server on (default: 8000).
    """
    server_address = ('', port)
    httpd = HTTPServer(server_address, LettersToSantaHandler)
    print(f"Server running on http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == '__main__':
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Invalid port number, using default 8000")
    
    run_server(port)

