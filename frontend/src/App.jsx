import React, { useState, useEffect } from "react";

// sample data
const sampleMessageHistory = ['Hi', 'Nice to meet you', 'Nice to meet you too!', 'How are you?']
const sampleRoleHistory = ['A', 'B', 'A', 'A']

// error message
const AT_LEAST_TWO_MESSAGES = "Please enter at least two messages to evaluate."
const INTERNAL_SERVER_ERROR = "Internal server error. Please try again later."

export default function ChatApp() {

  // current input
  const [role, setRole] = useState("A")
  const [message, setMessage] = useState("")

  // chatHistory
  const [messageHistory, setMessageHistory] = useState(sampleMessageHistory)
  const [roleHistory, setRoleHistory] = useState(sampleRoleHistory)
  const [words, setWords] = useState([])
  const [sentenceLabels, setSentenceLabels] = useState([])

  // labels 
  const [label, setLabel] = useState(undefined)
  const [explanation, setExplanation] = useState([])

  // error message
  const [showError, setShowError] = useState(false)
  const [hasInternalError, setHasInternalError] = useState(false)

  // loading state
  const [loading, setLoading] = useState(false)

  const handleSubmit = () => {
    if (messageHistory.length <= 2) {
      setShowError(true)
      return;
    }
    setShowError(false);
    setLoading(true);
    setLabel(undefined);
    setWords([]);
    setSentenceLabels([]);
    setHasInternalError(false);
    const endpoint = import.meta.env.MODE === "development"
      ? "http://localhost:8000/evaluate"
      : "https://friendzone-backend.nknguyenhc.net/evaluate"
    const conversation = [];
    for (let i = 0; i < messageHistory.length; i++) {
      conversation.push({
        role: roleHistory[i],
        response: messageHistory[i],
      });
    }
    fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        conversation: conversation,
      }),
    })
      .then(res => res.json())
      .then(data => {
        setLabel(data.label);
        setSentenceLabels(data.sequence_explanation);
        setWords(data.word_explanation
          .filter(([word, value]) => value < 0)
          .map(([word]) => word));
        setExplanation(data.sentence_explanation.map(([sentence, _]) => sentence));
      })
      .catch(err => {
        console.error(err);
        setHasInternalError(true);
      })
      .finally(() => {
        setLoading(false);
      });
  }

  return (
    <div style={{
      padding: "20px",
    }}>
      <h1>Lover or Friend Chat</h1>

      {/* Chat History with hightlight of important words*/}
      <div>
        <ul>
          {messageHistory.map((msg, idx) => 
            <li key={idx}>{roleHistory[idx]}: 
              <HighlightedSentence sentence={msg} explanationWords={words} label={label}></HighlightedSentence>
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
          setMessage('');
        }}
      >
        Send
      </button>

      {/* Label */}
      <div>
        {label !== undefined && <strong>Dialogue Label: </strong> }{numericLabelToString(label)}
      </div>

      {/* Label Explanation */}
      <div>
        {label !== undefined && (
          <div>
            <div>Which sentences are evidence for the label:</div>
            {explanation.map((sentence, idx) => (
              <div key={idx}>
                {sentence.role}: {sentence.response}
              </div>
            ))}
          </div>
        )}
      </div>
      
      {/* ReLabel Button */}
      <div>
        <button onClick={handleSubmit}>Are we friends or lovers?</button>
      </div>

      {/* Error Message */}
      <div style={{ color: "red" }}>
        {showError ? AT_LEAST_TWO_MESSAGES : null}
      </div>
      <div style={{ color: "red" }}>
        {hasInternalError ? INTERNAL_SERVER_ERROR : null}
      </div>
      
      {/* Recall Button */}
      <div>
        <button onClick={() => {
          setMessageHistory(messageHistory.slice(0, -1))
          setRoleHistory(roleHistory.slice(0, -1))
        }}>Remove last sentence</button>
      </div>

      {/* Loading State */}
      {loading && <Loader />}
    </div>
  );
}

function numericLabelToString(label) {
  switch (label) {
    case 0:
      return "Friend - Romantic";
    case 1:
      return "Romantic - Romantic";
    case -1:
      return "Friend - Friend";
  }
}

function HighlightedSentence({ sentence, explanationWords, label }) {
  const words = sentence.split(' ');

  return (
    <div>
      {words.map((word, index) => {
        const trimmedWord = word.replace(/^[.,;:!?]+|[.,;:!?]+$/g, ''); // Remove punctuation only at the start and end
        const isHighlighted = (label !== undefined) && explanationWords.includes(trimmedWord);
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
  return label !== "" && sentenceLabel !== undefined
    ? <span><strong> (Label so far: {numericLabelToString(sentenceLabel)})</strong></span>
    : null
}

function Loader() {
  return (
    <div className="loader-container">
      <div className="loader-text">Loading...</div>
    </div>
  );
}
