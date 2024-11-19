import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { TextField, Typography, Box, CircularProgress, IconButton, InputAdornment } from '@mui/material';
import { Mic, Stop } from '@mui/icons-material';

const VoiceChatBot = () => {
  const [inputText, setInputText] = useState('');
  const [conversation, setConversation] = useState([]);
  const [loading, setLoading] = useState(false);
  const [listening, setListening] = useState(false);

  const recognitionRef = useRef(null);
  const timeoutRef = useRef(null);
  const conversationEndRef = useRef(null);

  useEffect(() => {
    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.onresult = handleSpeechResult;
    }

    const storedConversation = JSON.parse(localStorage.getItem('conversation')) || [];
    setConversation(storedConversation);
  }, []);

  useEffect(() => {
    localStorage.setItem('conversation', JSON.stringify(conversation));
    scrollToBottom();
  }, [conversation]);

  const scrollToBottom = () => {
    conversationEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSpeechResult = (event) => {
    let interimTranscript = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        setInputText((prev) => prev + transcript);
      } else {
        interimTranscript += transcript;
      }
    }

    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => {
      if (inputText.trim()) {
        handleSubmit();
      }
    }, 500);

    setInputText((prev) => prev + interimTranscript);
  };

  const startListening = () => {
    if (recognitionRef.current) {
      recognitionRef.current.start();
      setListening(true);
    }
  };

  const stopListening = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      setListening(false);
    }
  };

  const handleSubmit = useCallback(async () => {
    if (!inputText.trim()) return;
    const promptText = inputText;
    setInputText('');
    setLoading(true);

    try {
      const response = await axios.post(process.env.REACT_APP_API_URL, {
        model: process.env.REACT_APP_DEFAULT_MODEL,
        prompt: promptText,
        stream: false,
      });

      const fullResponse = response.data.response || response.data;
      if (!fullResponse) {
        console.error('No response text found in API data.');
        return;
      }

      const newTextObj = { prompt: promptText, response: '', date: new Date().toISOString() };
      setConversation((prev) => [...prev, newTextObj]);

      let i = 0;
      const intervalId = setInterval(() => {
        setConversation((prev) => {
          const updatedConversation = [...prev];
          updatedConversation[updatedConversation.length - 1].response = fullResponse.substring(0, i);
          return updatedConversation;
        });
        i += 1;
        if (i > fullResponse.length) clearInterval(intervalId);
      }, 50);
    } catch (error) {
      console.error('Error fetching response:', error);
    } finally {
      setLoading(false);
    }
  }, [inputText, conversation]);

  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      handleSubmit();
    }
  };

  return (
    <Box sx={{ padding: '20px', maxWidth: '600px', margin: 'auto' }}>
      <Typography variant="h4" gutterBottom>
        Voice-based Ollama Chatbot
      </Typography>

      <Box sx={{ display: 'flex', alignItems: 'center', marginBottom: '20px' }}>
        <TextField
          label="Listening..."
          fullWidth
          variant="outlined"
          value={inputText}
          onKeyPress={handleKeyPress}
          InputProps={{
            readOnly: true,
            endAdornment: (
              <InputAdornment position="end">
                <IconButton onClick={listening ? stopListening : startListening}>
                  {listening ? <Stop color="secondary" /> : <Mic color="primary" />}
                </IconButton>
              </InputAdornment>
            ),
          }}
        />
      </Box>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', margin: '20px 0' }}>
          <CircularProgress />
        </Box>
      )}

      <Box sx={{ marginTop: '20px', maxHeight: '400px', overflowY: 'auto' }}>
        {conversation.map((textObj, index) => (
          <Box key={index} sx={{ marginBottom: '20px' }}>
            <Typography variant="body1" style={{ whiteSpace: 'pre-line' }}>
              <strong>Prompt:</strong> {textObj.prompt}
            </Typography>
            <Typography variant="body1" style={{ whiteSpace: 'pre-line', marginTop: '10px' }}>
              <strong>Response:</strong> {textObj.response}
            </Typography>
          </Box>
        ))}
        <div ref={conversationEndRef} />
      </Box>
    </Box>
  );
};

export default VoiceChatBot;
