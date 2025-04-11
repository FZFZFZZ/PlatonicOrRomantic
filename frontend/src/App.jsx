import React, { useState, useEffect } from "react";


export default function ChatApp() {

  // sample data
  let sampleMessageHistory = ['Hi', 'Nice to meet you', 'Nice to meet you too!', 'How are you?']
  let sampleRoleHistory = ['A', 'B', 'A', 'A']
  let sampleHighlightIndices = [[], [0, 3], [0, 3, 4], []]
  let sampleLabel = 'You are friends!'
  let sampleExplanation = 'because you don\'t sound familiar'
  let sampleSentenceLabels = [0, 0, 0, 1]

  // current input
  const [role, setRole] = useState("A")
  const [message, setMessage] = useState("")

  // chatHistory
  const [messageHistory, setMessageHistory] = useState(sampleMessageHistory)
  const [roleHistory, setRoleHistory] = useState(sampleRoleHistory)
  const [highlightIndices, setHighlightIndices] = useState(sampleHighlightIndices)
  const [sentenceLabels, setSentenceLabels] = useState(sampleSentenceLabels)

  // labels 
  const [label, setLabel] = useState(sampleLabel)
  const [explanation, setExplanation] = useState(sampleExplanation)

  useEffect(() => {
    if (label !== "") {
      setHighlightIndices(getHighlightIndices())
      setExplanation(getExplanation())
      setSentenceLabels(getSentenceLabels())
    } else {
      setExplanation("")
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
        {label === "" ? null : <strong>Dialogue Label: </strong> }{label}
      </div>
      <div>
        {label === "" ? null : <strong>Explanation: </strong> }{explanation}
      </div>
      
      {/* ReLabel Button */}
      <div>
        <button onClick={() => {
          setLabel((label) => (label === "" ? getLabel() : ""))
        }}>{label === "" ? 'Are we friends or lovers?' : 'Hide labels'}</button>
      </div>
      
      {/* Recall Button */}
      <div>
        <button onClick={() => {
          setMessageHistory(messageHistory.slice(0, -1))
          setRoleHistory(roleHistory.slice(0, -1))
          setSentenceLabels(sentenceLabels.slice(0, -1))
        }}>Recall</button>
      </div>
    </div>
  );
}


function handleSubmit() {

}

function getLabel() {
  const sampleLabel = 'You are lovers!'
  return sampleLabel
}


function getHighlightIndices() {
  const sampleHighlightIndices = [[], [0, 2], [0, 3, 4], []]
  return sampleHighlightIndices
}

function getExplanation() {
  const sampleExplanation = 'because you sound crazy.'
  return sampleExplanation
}

function getSentenceLabels() {
  const sampleSentenceLabels = [1, 0, 1, 0]
  return sampleSentenceLabels
}

function HighlightedSentence({ sentence, indices, label }) {
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

function SentenceLabelComponent( {sentenceLabel, label} ) {
  return label !== "" ? <span><strong> (Label: {sentenceLabel})</strong></span> : null
}
