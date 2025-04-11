import React, { useState, useEffect } from "react";


export default function ChatApp() {

  // sample data
  let sampleMessageHistory = ['Hi', 'Nice to meet you', 'Nice to meet you too!', 'How are you?']
  let sampleRoleHistory = ['A', 'B', 'A', 'A']
  let sampleHighlightIndices = [[], [0, 3], [0, 3, 4], []]
  let sampleLabel = 'You are friends!'
  let sampleExplanation = 'because you don\'t sound familiar'

  // current input
  const [role, setRole] = useState("A")
  const [message, setMessage] = useState("")

  // chatHistory
  const [messageHistory, setMessageHistory] = useState(sampleMessageHistory)
  const [roleHistory, setRoleHistory] = useState(sampleRoleHistory)
  const [highlightIndices, setHighlightIndices] = useState(sampleHighlightIndices)

  // labels 
  const [label, setLabel] = useState(sampleLabel)
  const [explanation, setExplanation] = useState(sampleExplanation)

  useEffect(() => {
    if (label === "") {
      setHighlightIndices(getHighlightIndices())
      setExplanation(getExplanation)
    }
  }, [label])

  return (
    <div>
      <h1>Lover or Friend Chat</h1>

      {/* Chat History with hightlight of important words*/}
      <div>
        <ul>
          {messageHistory.map((msg, idx) => 
            <li>{roleHistory[idx]}: 
              <SentenceHighlight sentence={msg} indices={highlightIndices[idx]} label={label}></SentenceHighlight>
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
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={message}
          onChange={(e) => {
            setMessage(e.target.value);
          }}
        />
        <button
          type="submit"
          onSubmit={() => {
            setMessageHistory((messageHistory) => [...messageHistory, message])
            setRoleHistory((roleHistory) => [...roleHistory, role])
            setHighlightIndices((highlightIndices) => [...highlightIndices, []])
          }}
        >
          Send
        </button>
      </form>

      {/* Label */}
      <div>
        <strong>Dialogue Label: </strong> {label}
      </div>
      <div>
        <strong>Explanation: </strong> {explanation}
      </div>
      
      {/* ReLabel Button */}
      <div>
        <button onClick={() => {
          setLabel((label) => (label === "" ? getLabel() : ""))
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


function getHighlightIndices() {
  const sampleHighlightIndices = [[], [0, 2], [0, 3, 4], []]
  return sampleHighlightIndices
}

function getExplanation() {
  const sampleExplanation = 'because yaa don\'t sound familiar'
  return sampleExplanation
}

function SentenceHighlight({ sentence, indices, label }) {
  const words = sentence.split(' ');

  return (
    <div>
      {words.map((word, index) => {
        const isHighlighted = (label !== "") && indices.includes(index);
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