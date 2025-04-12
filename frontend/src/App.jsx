import React, { useState, useEffect } from "react";

// sample data
const sampleMessageHistory = ['Hi', 'Nice to meet you', 'Nice to meet you too!', 'How are you?']
const sampleRoleHistory = ['A', 'B', 'A', 'A']

export default function ChatApp() {

  // current input
  const [role, setRole] = useState("A")
  const [message, setMessage] = useState("")

  // chatHistory
  const [messageHistory, setMessageHistory] = useState(sampleMessageHistory)
  const [roleHistory, setRoleHistory] = useState(sampleRoleHistory)
  const [highlightIndices, setHighlightIndices] = useState([])
  const [sentenceLabels, setSentenceLabels] = useState([])

  // labels 
  const [label, setLabel] = useState('')

  return (
    <div style={{
      padding: "20px",
    }}>
      <h1>Lover or Friend Chat</h1>

      {/* Chat History with hightlight of important words*/}
      <div>
        <ul>
          {messageHistory.map((msg, idx) => 
            <li>{roleHistory[idx]}: 
              <HighlightedSentence sentence={msg} indices={highlightIndices[idx]} label={label}></HighlightedSentence>
              <SentenceLabelComponent sentenceLabel={sentenceLabels[idx]} label={label}></SentenceLabelComponent>
            </li>)}
        </ul>
      </div>

      {/* Role Switcher */}
      <div>
        <button
          onClick={() => {
            setRole((prev) => (prev === "A" ? "B" : "A"));
          }}
        >
          Current Role: {role} (Click to switch)
        </button>
      </div>

      {/* Message Input */}
      <input
        type="text"
        value={message}
        onChange={(e) => {
          setMessage(e.target.value);
        }}
      />
      <button
        type="submit"
        onClick={() => {
          setMessageHistory((messageHistory) => [...messageHistory, message])
          setRoleHistory((roleHistory) => [...roleHistory, role])
          setHighlightIndices((highlightIndices) => [...highlightIndices, []])
        }}
      >
        Send
      </button>

      {/* Label */}
      <div>
        {label === "" ? null : <strong>Dialogue Label: </strong> }{label}
      </div>
      
      {/* ReLabel Button */}
      <div>
        <button onClick={() => {
          setLabel((label) => label)
        }}>Are we friends or lovers?</button>
      </div>
      
      {/* Recall Button */}
      <div>
        <button onClick={() => {
          setMessageHistory(messageHistory.slice(0, -1))
          setRoleHistory(roleHistory.slice(0, -1))
          setSentenceLabels(sentenceLabels.slice(0, -1))
        }}>Undo</button>
      </div>
    </div>
  );
}


function handleSubmit() {

}

function HighlightedSentence({ sentence, indices, label }) {
  const words = sentence.split(' ');

  return (
    <div>
      {words.map((word, index) => {
        const isHighlighted = (label !== "") && indices?.includes(index);
        return (
          <span
            key={index}
            style={{
              backgroundColor: isHighlighted ? 'yellow' : 'transparent',
              padding: '0 2px',
              marginRight: '4px',
            }}
          >
            {word} 
          </span>
        );
      })}
    </div>
  );
};

function SentenceLabelComponent( {sentenceLabel, label} ) {
  return label !== "" ? <span><strong> (Label: {sentenceLabel})</strong></span> : null
}
