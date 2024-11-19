// src/api/ollamaApi.js
import axios from 'axios';

export const generateResponse = async (model, prompt, images) => {
  const response = await axios.post('http://localhost:11434/api/generate', {
    model,
    prompt,
    stream: false,
    images
  });
  return response.data.response;
};
