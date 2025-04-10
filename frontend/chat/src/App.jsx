import React, { useState, useEffect } from "react";

import DialogLabel, { DetailedChatHistory } from "./DialogLabel";
import { SentenceHighlight } from "./hightlight";

// To be replaced with a query
function getWordIndices(chatHistory) {
  // Just highlight 0-th word of 0-th sentence
  // and first two words of 1-st sentence for now
  let indices = {};
  indices[0] = [0];
  indices[1] = [0, 1]
  return indices
}

export default function ChatApp() {

  const [role, setRole] = useState("A");
  const [message, setMessage] = useState("");
  const [chatHistory, setChatHistory] = useState(() => {
    return JSON.parse(localStorage.getItem("chatHistory")) || []
  });
  const [relabelTrigger, setRelabelTrigger] = useState(0);
  const [suppressTrigger, setSuppressTrigger] = useState(0);
  const [wordIndices, setWordIndices] = useState({})
  
  useEffect(() => {
    localStorage.setItem("chatHistory", JSON.stringify(chatHistory));
  }, [chatHistory]);

  useEffect(() => {
    const indices = getWordIndices(chatHistory);
    setWordIndices(indices);
  }, [relabelTrigger])


  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim() === "") return;

    const newEntry = { role, message };
    setChatHistory([...chatHistory, newEntry]);
    setMessage("");
  };

  const toggleRole = () => {
    setRole((prev) => (prev === "A" ? "B" : "A"));
  };

  return (
    <div className="p-4 max-w-md mx-auto">
      <h1 className="text-2xl font-bold mb-4">Lover or Friend Chat</h1>

      {/* Chat History with hightlight of important words*/}
      <DetailedChatHistory 
        chatHistory={chatHistory}
        trigger={relabelTrigger}
        suppress={suppressTrigger}
      />

      {/* Role Switcher */}
      <div className="mb-4">
        <button
          onClick={toggleRole}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Current Role: {role} (Click to switch)
        </button>
      </div>

      {/* Message Input */}
      <form onSubmit={handleSubmit} className="mb-4">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Enter your message"
          className="w-full p-2 border rounded mb-2"
        />
        <button
          type="submit"
          className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
        >
          Send
        </button>
      </form>

      {/* Label */}
      <DialogLabel 
        chatHistory={chatHistory} 
        trigger={relabelTrigger}
        suppress={suppressTrigger}
      />

      {/* ReLabel Button */}
      <button
        onClick={() => {
          setRelabelTrigger(prev => (prev + 1));
          setSuppressTrigger(1);
        }}
        className="px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 mb-4"
      >
        Are we friends or lovers?
      </button>
    </div>
  );
}
