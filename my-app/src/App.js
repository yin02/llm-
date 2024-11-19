import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { TextField, Typography, Box, List, ListItem, ListItemText, Divider, IconButton, Collapse, InputAdornment, Drawer, Select, MenuItem, FormControl, InputLabel, Toolbar, AppBar, Button, CircularProgress } from '@mui/material';
import { ExpandLess, ExpandMore, Menu, ChevronLeft, ChevronRight, Delete, Image, Mic } from '@mui/icons-material';

const OllamaVoiceApi = () => {
  const [inputText, setInputText] = useState('');
  const [conversation, setConversation] = useState([]);
  const [previousTexts, setPreviousTexts] = useState([]);
  const [model, setModel] = useState('qwen2.5');
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [expandedDates, setExpandedDates] = useState({});
  const [loading, setLoading] = useState(false); // 用于显示加载状态
  const [modelMessage, setModelMessage] = useState(''); // 模型切换提示信息
  const [listening, setListening] = useState(false);
  const [recognition, setRecognition] = useState(null);
  const [tempTranscript, setTempTranscript] = useState('');
  const [submitTimeout, setSubmitTimeout] = useState(null);

  useEffect(() => {
    const storedTexts = JSON.parse(localStorage.getItem('previousTexts')) || [];
    const recentTexts = storedTexts.filter(textObj => {
      const date = new Date(textObj.date);
      const today = new Date();
      return (today - date) / (1000 * 60 * 60 * 24) <= 20;
    });
    setPreviousTexts(recentTexts);
  }, []);

  useEffect(() => {
    localStorage.setItem('previousTexts', JSON.stringify(previousTexts));
  }, [previousTexts]);

  useEffect(() => {
    if ('webkitSpeechRecognition' in window) {
      const SpeechRecognition = window.webkitSpeechRecognition;
      const newRecognition = new SpeechRecognition();
      newRecognition.continuous = true;
      newRecognition.interimResults = true;
      newRecognition.lang = 'en-US';

      newRecognition.onresult = (event) => {
        const lastResult = event.results[event.results.length - 1];
        if (lastResult.isFinal) {
          setInputText((prev) => prev + ' ' + lastResult[0].transcript);
          resetSubmitTimeout();
        } else {
          setTempTranscript(lastResult[0].transcript);
        }
      };

      newRecognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
      };

      newRecognition.onend = () => {
        if (listening) {
          newRecognition.start(); // Restart recognition to keep it continuous
        }
      };

      setRecognition(newRecognition);
    } else {
      console.warn('Web Speech API is not supported in this browser.');
    }
  }, [listening]);

  const toggleListening = () => {
    if (listening) {
      recognition.stop();
      setListening(false);
    } else {
      recognition.start();
      setListening(true);
    }
  };

  const handleInputChange = (event) => {
    setInputText(event.target.value);
    resetSubmitTimeout();
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      handleSubmit();
    }
  };

  const handleModelChange = (event) => {
    setModel(event.target.value);
    setModelMessage(`Using model: ${event.target.value}`); // 更改模型时更新提示信息
  };

  const resetSubmitTimeout = () => {
    if (submitTimeout) clearTimeout(submitTimeout);
    setSubmitTimeout(setTimeout(() => {
      if (inputText.trim() !== '') {
        handleSubmit();
      }
    }, 1000));
  };

  const handleSubmit = async () => {
    if (inputText.trim() === '') return;

    const promptText = inputText;
    setInputText(''); // 清空输入框
    setLoading(true); // 开始加载

    // 构建完整的 prompt，包括当前对话的所有内容
    const fullPrompt = conversation.reduce((acc, textObj) => {
      return `${acc}
User: ${textObj.prompt}
Model: ${textObj.response}`;
    }, '') + `
User: ${promptText}`;

    try {
      const response = await axios.post('http://localhost:11434/api/generate', {
        model,
        prompt: fullPrompt,
        stream: false,
      });
      const fullResponse = response.data.response;
      const newTextObj = { prompt: promptText, response: fullResponse, date: new Date().toISOString() };
      setPreviousTexts(prev => {
        const today = new Date().toISOString().split('T')[0];
        let updatedTexts = prev.map(textObj => {
          if (textObj.date.split('T')[0] === today) {
            return { ...textObj, responses: [...textObj.responses, newTextObj] };
          }
          return textObj;
        });
        if (!updatedTexts.some(textObj => textObj.date.split('T')[0] === today)) {
          updatedTexts.push({ date: new Date().toISOString(), responses: [newTextObj] });
        }
        if (updatedTexts.length > 20) {
          updatedTexts = updatedTexts.slice(updatedTexts.length - 20);
        }
        return updatedTexts;
      });
      setConversation((prev) => [...prev, newTextObj]);
    } catch (error) {
      console.error('Error fetching response:', error);
    } finally {
      setLoading(false); // 结束加载
    }
  };

  const startNewConversation = () => {
    setConversation([]);
  };

  const toggleDrawer = () => {
    setDrawerOpen(!drawerOpen);
  };

  const toggleDate = (date) => {
    setExpandedDates(prev => ({ ...prev, [date]: !prev[date] }));
  };

  const handleHistoryClick = (textObj) => {
    setConversation(textObj.responses);
  };

  const handleDeleteResponse = (date, index) => {
    setPreviousTexts(prev => {
      return prev.map(textObj => {
        if (textObj.date === date) {
          const updatedResponses = textObj.responses.filter((_, i) => i !== index);
          return { ...textObj, responses: updatedResponses };
        }
        return textObj;
      });
    });
  };

  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1, backgroundColor: 'white', color: 'black' }}>
        <Toolbar>
          <IconButton color="inherit" edge="start" onClick={toggleDrawer} sx={{ mr: 2 }}>
            {drawerOpen ? <ChevronLeft /> : <Menu />}
          </IconButton>
          <Typography variant="h6" noWrap>
            History
          </Typography>
          <FormControl variant="outlined" sx={{ ml: 2, minWidth: 120 }}>
            <InputLabel>Model</InputLabel>
            <Select value={model} onChange={handleModelChange} label="Model">
              <MenuItem value="qwen2.5">qwen2.5</MenuItem>
              <MenuItem value="llama3.2:latest">gemma2:latest</MenuItem>
              <MenuItem value="qwen2:latest">qwen2:latest</MenuItem>
            </Select>
          </FormControl>
        </Toolbar>
      </AppBar>
      <Drawer
        variant="persistent"
        anchor="left"
        open={drawerOpen}
        sx={{
          width: drawerOpen ? 240 : 0,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: 240,
            boxSizing: 'border-box',
          },
        }}
      >
        <Toolbar />
        <Divider />
        <Button onClick={startNewConversation} variant="contained" color="primary" sx={{ m: 2 }}>
          New Conversation
        </Button>
        <List>
          {previousTexts.map((textObj, index) => (
            <React.Fragment key={index}>
              <ListItem button onClick={() => toggleDate(textObj.date)}>
                <ListItemText primary={new Date(textObj.date).toLocaleDateString()} />
                {expandedDates[textObj.date] ? <ExpandLess /> : <ExpandMore />}
              </ListItem>
              <Collapse in={expandedDates[textObj.date]} timeout="auto" unmountOnExit>
                {textObj.responses.map((response, subIndex) => (
                  <List key={subIndex}>
                    <ListItem alignItems="flex-start" button>
                      <ListItemText
                        primary={response.prompt}
                        secondary={
                          <>
                            <Typography component="span" variant="body2" color="textPrimary">
                              {response.response}
                            </Typography>
                            <br />
                            <Typography component="span" variant="caption" color="textSecondary">
                              {new Date(response.date).toLocaleTimeString()}
                            </Typography>
                          </>
                        }
                        onClick={() => handleHistoryClick(textObj)}
                      />
                      <IconButton edge="end" aria-label="delete" onClick={() => handleDeleteResponse(textObj.date, subIndex)}>
                        <Delete />
                      </IconButton>
                    </ListItem>
                    <Divider component="li" />
                  </List>
                ))}
              </Collapse>
            </React.Fragment>
          ))}
        </List>
      </Drawer>
      {!drawerOpen && (
        <IconButton onClick={toggleDrawer} style={{ position: 'fixed', top: 10, left: 10 }}>
          <ChevronRight />
        </IconButton>
      )}
      <Box component="main" sx={{ flexGrow: 1, p: 3, display: 'flex', flexDirection: 'column', height: '100vh' }}>
        <Toolbar />
        <Box sx={{ flex: 1, overflow: 'auto' }}>
          {modelMessage && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="body1" color="textSecondary">
                {modelMessage}
              </Typography>
            </Box>
          )}
          {conversation.map((textObj, index) => (
            <Box key={index} sx={{ marginBottom: '20px' }}>
              <Typography variant="body1" style={{ whiteSpace: 'pre-line', wordWrap: 'break-word' }}>
                <strong>Prompt:</strong> {textObj.prompt}
              </Typography>
              <Typography variant="body1" style={{ whiteSpace: 'pre-line', wordWrap: 'break-word', marginTop: '10px' }}>
                <strong>Response:</strong> {textObj.response}
              </Typography>
            </Box>
          ))}
          {loading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
              <CircularProgress />
            </Box>
          )}
        </Box>
        <Box sx={{ position: 'sticky', bottom: 0, backgroundColor: 'white', padding: '10px', borderTop: '1px solid #ddd', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <TextField
            label="Enter your prompt"
            fullWidth
            variant="outlined"
            value={inputText}
            onChange={handleInputChange}
          />
        </Box>
      </Box>
      <IconButton onClick={toggleListening} color={listening ? 'primary' : 'default'} sx={{ position: 'fixed', right: 20, bottom: 20 }}>
        <Mic />
      </IconButton>
      {tempTranscript && (
        <Box sx={{ position: 'fixed', right: 20, bottom: 80, backgroundColor: 'rgba(255, 255, 255, 0.8)', padding: 1, borderRadius: 1 }}>
          <Typography variant="body1" color="textSecondary">
            {tempTranscript}
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default OllamaVoiceApi;
