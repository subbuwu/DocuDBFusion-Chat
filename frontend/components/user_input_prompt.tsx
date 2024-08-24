import { Input } from "@/components/ui/input";
import { Button } from "./ui/button";
import { Send, MessageSquare } from "lucide-react";
import { motion } from "framer-motion";

interface ChatInputProps {
  value: string;
  onChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onSubmit: () => void;
  isLoading: boolean;
}

export const ChatInput = ({ value, onChange, onSubmit, isLoading }: ChatInputProps) => (
  <div className="fixed bottom-0 left-0 right-0 p-4 bg-transparent">
    <div className="max-w-3xl mx-auto bg-white rounded-lg shadow-lg">
      <div className="relative">
        <Input
          placeholder="Ask me something"
          value={value}
          onChange={onChange}
          className="w-full pr-12 py-6 focus:ring-2 text-base focus:ring-blue-400 focus:border-blue-400 border-none shadow-none outline-none transition duration-200 ease-in-out lg:h-16"
        />
        <Button 
          type="submit" 
          onClick={onSubmit} 
          disabled={isLoading}
          className="absolute right-2 top-1/2 transform -translate-y-1/2 p-1 bg-transparent hover:bg-transparent"
        >
          {isLoading ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-blue-500"
            />
          ) : (
            <Send className="h-5 w-5 text-blue-500" />
          )}
        </Button>
      </div>
    </div>
  </div>
);