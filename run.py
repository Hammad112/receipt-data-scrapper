"""
Simple launcher for Receipt Intelligence System.
This runs the Streamlit UI which handles everything internally.
"""

import os
import sys
import subprocess

def main():
    print(" RECEIPT INTELLIGENCE SYSTEM")
    print("=" * 60)
    print()
    print(" The system will:")
    print("  Check/create Pinecone index")
    print("  Process receipts if needed")
    print("  Launch web interface")
    print()
    print(" Starting Streamlit UI...")
    print(" Will open at: http://localhost:8501")
    print(" Press Ctrl+C to stop")
    print("=" * 60)
    print()
    
    streamlit_script = os.path.join(os.path.dirname(__file__), "src", "ui", "streamlit_app.py")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            streamlit_script
        ], check=True)
    except KeyboardInterrupt:
        print("\n\n Shutting down gracefully...")
    except FileNotFoundError:
        print("\n Streamlit not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"], check=True)
        print("\n Streamlit installed. Please run again: python run.py")
    except Exception as e:
        print(f"\n Error: {e}")
        print("\n Try running directly:")
        print(f"   streamlit run {streamlit_script}")

if __name__ == "__main__":
    main()
