#!/usr/bin/env python3
"""Fix SSL certificate issues on macOS for Python.

This script helps resolve SSL certificate verification errors when accessing
Google Cloud Storage or other HTTPS services.

Usage:
    python fix_ssl_macos.py
"""

import os
import ssl
import certifi
import subprocess
import sys


def install_certificates():
    """Install certificates for Python on macOS."""
    print("Installing certificates for Python on macOS...")
    
    # Get the path to the Python installation
    python_path = sys.executable
    python_dir = os.path.dirname(os.path.dirname(python_path))
    
    # Look for the Install Certificates command
    install_cert_script = os.path.join(python_dir, "Install Certificates.command")
    
    if os.path.exists(install_cert_script):
        print(f"Running: {install_cert_script}")
        subprocess.run(["/bin/bash", install_cert_script], check=True)
        print("Certificates installed successfully!")
    else:
        print("Install Certificates.command not found.")
        print("Trying alternative method...")
        
        # Alternative: Set SSL_CERT_FILE environment variable
        cert_file = certifi.where()
        print(f"Setting SSL_CERT_FILE to: {cert_file}")
        
        # Create a shell script to set the environment variable
        shell_script = """
# Add this to your ~/.bashrc, ~/.zshrc, or ~/.bash_profile
export SSL_CERT_FILE="{}"
export REQUESTS_CA_BUNDLE="{}"
""".format(cert_file, cert_file)
        
        print("\nAdd the following to your shell configuration file:")
        print(shell_script)
        
        # Also set it for the current session
        os.environ['SSL_CERT_FILE'] = cert_file
        os.environ['REQUESTS_CA_BUNDLE'] = cert_file
        
        print("\nEnvironment variables set for current session.")


def test_ssl_connection():
    """Test SSL connection to Google Cloud Storage."""
    import urllib.request
    
    print("\nTesting SSL connection to storage.googleapis.com...")
    try:
        response = urllib.request.urlopen('https://storage.googleapis.com/')
        print("✓ SSL connection successful!")
        return True
    except Exception as e:
        print(f"✗ SSL connection failed: {e}")
        return False


def main():
    print("SSL Certificate Fix for macOS")
    print("=" * 50)
    
    # Install certificates
    install_certificates()
    
    # Test the connection
    if test_ssl_connection():
        print("\nSSL certificates are properly configured!")
    else:
        print("\nSSL issues persist. You may need to:")
        print("1. Restart your terminal")
        print("2. Run: pip install --upgrade certifi")
        print("3. Reinstall Python from python.org")


if __name__ == "__main__":
    main()