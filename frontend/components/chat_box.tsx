import { User } from "lucide-react"

export const ChatMessage = ({ isUser, content } : { isUser : boolean , content : string }) => (
    <div className={`flex items-start space-x-2 sm:space-x-4 ${isUser ? 'justify-end' : ''}`}>
      {!isUser && (
        <div className="w-10 h-10 rounded-full text-sm bg-purple-500 flex items-center justify-center text-white">
          Evva
        </div>
      )}
      <div className={`p-2 sm:p-4 rounded-lg text-sm sm:text-base shadow max-w-[80%] ${isUser ? 'bg-blue-500' : 'bg-white'}`}>
        <p className={isUser ? 'text-white' : ''}>{content}</p>
      </div>
      {isUser && (
        <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center">
          <User className="h-5 w-5 text-gray-600" />
        </div>
      )}
    </div>
  )