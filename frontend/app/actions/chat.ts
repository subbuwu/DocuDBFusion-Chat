import axios from 'axios';

export const fetchChatResponse = async (input: string) => {
  try {
    const response = await axios.post('http://127.0.0.1:8000/chat', { question: input }, {
      headers: {
        'Content-Type': 'application/json',
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching chat response:', error);
    throw new Error('Could not fetch response.');
  }
};
