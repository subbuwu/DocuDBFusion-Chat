import { Bot, Database, Brain, Sparkles } from "lucide-react";
import { motion } from "framer-motion";
import "../app/globals.css"

export default function Navbar() {
  return (
    <motion.nav 
      className="bg-gray-800 shadow-lg"
      initial={{ opacity: 0, y: -50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          <motion.div 
            className="flex items-center space-x-4"
            initial={{ x: -50 }}
            animate={{ x: 0 }}
            transition={{ delay: 0.2, type: "spring", stiffness: 120 }}
          >
            <Bot className="h-10 w-10 text-white" />
            <h1 className="text-5xl font-bold tracking-wider text-white font-arrayfont">Evva Bot</h1>
          </motion.div>
          
          <motion.div 
            className="hidden md:flex items-center space-x-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.5 }}
          >
            <Database className="h-6 w-6 text-white" />
            <span className="text-white text-2xl font-bold font-arrayfont">SQLite Powered</span>
            <Brain className="h-6 w-6 text-white" />
            <span className="text-white text-2xl font-bold font-arrayfont">LLM Enhanced</span>
          </motion.div>
        </div>
      </div>
      
      {/* <motion.div 
        className="bg-[#16f7c6] p-2 text-center"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7, duration: 0.5 }}
      >
        <p className="text-gray-800 font-medium italic">
          "I'm not just any bot, I'm the cream of the crop... or at least the whey protein of the digital smoothie!"
        </p>
      </motion.div> */}
      
      <motion.div 
        className="absolute top-0 left-0 w-full h-1 bg-white"
        initial={{ scaleX: 0 }}
        animate={{ scaleX: 1 }}
        transition={{ duration: 1, ease: "easeInOut" }}
      />
    
    </motion.nav>
  );
}