#!/usr/bin/env python3
"""
Simple test client for the voice assistant WebSocket.
Sends a WAV file, prints transcript and reply, saves response audio to a file.

Usage:
  # Record a WAV (e.g. with Audacity or: sox -d -d trim 0 5 test.wav) then:
  python client.py path/to/recording.wav

  # Or pipe raw bytes (e.g. WAV from stdin):
  python client.py --stdin < recording.wav
"""
import argparse
import asyncio
import json
import sys
from typing import Optional

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets")
    sys.exit(1)


async def run(url: str, audio_path: Optional[str], stdin: bool, out_audio_path: str):
    if stdin:
        audio_bytes = sys.stdin.buffer.read()
    elif audio_path:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
    else:
        print("Provide either an audio file path or --stdin")
        sys.exit(1)

    if not audio_bytes:
        print("No audio data")
        sys.exit(1)

    print(f"Connecting to {url} ...")
    async with websockets.connect(url) as ws:
        print("Sending audio (%d bytes) ..." % len(audio_bytes))
        await ws.send(audio_bytes)

        audio_chunks = []
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=120.0)
            except asyncio.TimeoutError:
                print("Timeout waiting for response")
                break

            if isinstance(msg, bytes):
                audio_chunks.append(msg)
                print("Received audio: %d bytes" % len(msg))
                continue

            data = json.loads(msg)
            kind = data.get("type")
            if kind == "transcript":
                print("Transcript:", data.get("text", ""))
            elif kind == "reply":
                print("Reply:", data.get("text", ""))
            elif kind == "done":
                print("Done.")
                break
            elif kind == "error":
                print("Error:", data.get("message", ""))
                break
            elif kind == "pong":
                pass
            else:
                print("Message:", data)

        if audio_chunks:
            out_bytes = b"".join(audio_chunks)
            with open(out_audio_path, "wb") as f:
                f.write(out_bytes)
            print("Saved response audio to %s (%d bytes)" % (out_audio_path, len(out_bytes)))


def main():
    p = argparse.ArgumentParser(description="Voice assistant test client")
    p.add_argument("audio_file", nargs="?", help="Path to WAV file to send")
    p.add_argument("--stdin", action="store_true", help="Read audio bytes from stdin")
    p.add_argument("--url", default="ws://localhost:8000/ws", help="WebSocket URL")
    p.add_argument("--out", default="response.mp3", help="Path to save response audio")
    args = p.parse_args()

    asyncio.run(run(args.url, args.audio_file, args.stdin, args.out))


if __name__ == "__main__":
    main()
