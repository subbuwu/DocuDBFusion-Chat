# PDF and SQLite Chat Application ([Watch the app in action](https://drive.google.com/file/d/1j6j5-GYDTaT_53lsi6ciUI8Xz-QwgslW/view?usp=sharing))

This full-stack application allows users to chat with PDF content and SQLite database information simultaneously using natural language queries.
## Architecture Overview

The application follows a client-server architecture:

1. **Frontend**: React/Next.js application for user interface
2. **Backend**: FastAPI server handling API requests, database queries, and PDF processing
3. **Database**: SQLite for storing survey data
4. **PDF Storage**: Local server directory or cloud storage (e.g., AWS S3)
5. **NLP Model**: OpenAI API or open-source alternative for natural language processing


Key components:
- `pages/`: Next.js page components
- `components/`: Reusable React components (e.g., ChatWindow, PDFUploader)
- `api/`: API client for communicating with the backend
Image :
<img width="1512" alt="Screenshot 2024-08-24 at 11 44 37 PM" src="https://github.com/user-attachments/assets/68ebcbbd-8ac2-42cc-b74b-b591c0ae351c">
<img width="319" alt="Screenshot 2024-08-24 at 11 54 30 PM" src="https://github.com/user-attachments/assets/617bd1a3-7651-47ff-9ed7-952ed2d477f7">

