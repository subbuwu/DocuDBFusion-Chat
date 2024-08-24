"use client";
import { useState, useEffect, useRef } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { UploadDocuments } from "@/components/upload_section";
import { ChatMessage } from "@/components/chat_box";
import { ChatInput } from "@/components/user_input_prompt";
import { motion, AnimatePresence } from "framer-motion";
import { fetchChatResponse } from "./actions/chat";
import { Toaster, toast } from 'sonner'
import Navbar from "@/components/navbar";

export default function Home() {
  const [input, setInput] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [messages, setMessages] = useState([{ isUser: false, content: "Hello! How can I assist you today?" }]);
  const [isLoading, setIsLoading] = useState(false);

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleFileChange = (file: File | null) => {
    if (file && file.type !== "application/pdf") {
      toast.error('Please upload a PDF File')
      setFile(null);
      return;
    }
    setFile(file);
  };


  const handleSubmit = async () => {
    if (!input.trim()) return;

    setMessages([...messages, { isUser: true, content: input }]);
    setInput("");
    setIsLoading(true);

    try {
      const { answer } = await fetchChatResponse(input);
      setMessages(prev => [...prev, { isUser: false, content: answer }]);
    } catch (error) {
      setMessages(prev => [...prev, { isUser: false, content: "Error: Could not fetch response." }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <Navbar />
      <Toaster position="top-center" richColors />
      <div className="flex-grow flex flex-col overflow-hidden">
        <UploadDocuments 
          onFileChange={handleFileChange} 
          file={file} 
        />
        <ScrollArea 
          ref={scrollRef} 
          className="flex-grow p-2 pt-4 sm:p-4 overflow-y-auto"
        >
          <div className="space-y-4 max-w-3xl mx-auto pb-24">
            <AnimatePresence>
              {messages.map((message, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 50 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -50 }}
                  transition={{ duration: 0.3 }}
                >
                  <ChatMessage isUser={message.isUser} content={message.content} />
                </motion.div>
              ))}
            </AnimatePresence>
            {isLoading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-start space-x-4"
              >
                <div className="w-8 h-8 rounded-full bg-purple-500 flex items-center justify-center text-white">C</div>
                <div className="p-3 rounded-lg shadow flex justify-center items-center bg-white">
                  <ThreeDotsLoader />
                </div>
              </motion.div>
            )}
          </div>
        </ScrollArea>
      </div>
      <ChatInput 
        value={input} 
        onChange={(e) => setInput(e.target.value)} 
        onSubmit={handleSubmit} 
        isLoading={isLoading} 
      />
    </div>
  );
}

const ThreeDotsLoader = () => (
  <div className="flex space-x-1">
    {[0, 1, 2].map((index) => (
      <motion.div
        key={index}
        className="w-2 h-2 bg-gray-500 rounded-full"
        initial={{ y: 0 }}
        animate={{ y: [0, -5, 0] }}
        transition={{ duration: 0.6, repeat: Infinity, repeatType: "loop", delay: index * 0.2 }}
      />
    ))}
  </div>
);