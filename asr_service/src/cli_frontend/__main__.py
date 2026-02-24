"""
Entry point for CLI frontend.

Run with: python -m cli_frontend
Or via Makefile: make run-cli
"""

from cli_frontend.app import MeetingScribeApp


def main():
    """Launch MeetingScribe CLI application."""
    app = MeetingScribeApp()
    app.run()


if __name__ == "__main__":
    main()
