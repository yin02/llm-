# Ollama Voice App

This is a voice-based chat application that uses Ollama API to generate responses based on user voice input.

## Features

- Real-time voice input using SpeechRecognition API
- Automatic response trigger after 0.6 seconds of silence
- Progressive response display (one character at a time)

## Setup

1. Install dependencies: `npm install`
2. Add `.env` file with your API URL.
3. Start the app: `npm start`

## Environment Variables

- `REACT_APP_API_URL` - The base URL for Ollama API.

## Technologies

- React
- Material UI
- Axios
