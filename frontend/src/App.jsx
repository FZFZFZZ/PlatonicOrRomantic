import React, { useState, useEffect } from "react";


export default function ChatApp() {

  // test
  const [click, setClick] = useState(0)

  // sample data
  let sampleMessageHistory = ['Hi', 'Nice to meet you', 'Nice to meet you too!', 'How are you?']
  let sampleRoleHistory = ['A', 'B', 'A', 'A']

  // current input
  const [role, setRole] = useState("A")
  const [message, setMessage] = useState("")

  // chatHistory
  const [messageHistory, setMessageHistory] = useState(sampleMessageHistory)
  const [roleHistory, setRoleHistory] = useState(sampleRoleHistory)
  const [importantIndex, setImportantIndex] = useState([])

  // labels 
  const [label, setLabel] = useState("")
  const [explanation, setExplanation] = useState("")



  return (
    <div>
      <h1>Lover or Friend Chat</h1>
      <h1>Click: {click}</h1>

      {/* Chat History with hightlight of important words*/}
      <div>
        <ul>
          {messageHistory}
          {roleHistory}
          {/* {messageHistory.map((msg, idx) => <li>{roleHistory[idx]}: {msg}</li>)} */}
        </ul>
      </div>

      {/* Role Switcher */}
      <div>
        <button
          onClick={() => {
            setRole((prev) => (prev === "A" ? "B" : "A"));
            setClick(click+1)
          }}
        >
          Current Role: {role} (Click to switch)
        </button>
      </div>

      {/* Message Input */}
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={message}
          onChange={(e) => {
            setMessage(e.target.value);
            setClick(click+1)
          }}
        />
        <button
          type="submit"
          onSubmit={() => {
            setMessageHistory(messageHistory.push(message))
            setRoleHistory(roleHistory.push(role))
          }}
        >
          Send
        </button>
      </form>

      {/* Label */}
      <div>
        <strong>Dialogue Label:</strong> {label}
      </div>
      
      {/* ReLabel Button */}
      <div>
        <button onClick={() => {
          setLabel((label) => (label === "" ? getLabel() : ""))
          setClick(click+1)
        }}>Are we friends or lovers?</button>
      </div>
      
      {/* Recall Button */}
      <div>
        <button onClick={() => {
          setMessageHistory(messageHistory.slice(0, -1))
          setRoleHistory(roleHistory.slice(0, -1))
        }}>Recall</button>
      </div>
    </div>
  );
}


function handleSubmit() {

}

function getLabel() {
  const sampleLabel = 'You are friends.'
  return sampleLabel
}

