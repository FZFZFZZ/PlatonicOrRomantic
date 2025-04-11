import React, { useEffect, useState } from "react";

import { SentenceHighlight } from "./hightlight";

// Dummy label function — replace with backend call later
function getDialogLabel(chatHistory) {
  console.log(chatHistory)
  // Just return "YYY" for now
  return "YYY" + chatHistory;
}

// To be replaced with a query
function getWordIndices(chatHistory) {
  // Just highlight 0-th word of 0-th sentence
  // and first two words of 1-st sentence for now
  let indices = {};
  indices[0] = [0];
  indices[1] = [0, 1]
  return indices
}

function getProgress(chatHistory) {
  console.log(chatHistory)
  return []
}

export function DetailedChatHistory({ chatHistory, trigger, suppress }) {
    const [wordIndices, setWordIndices] = useState([])
    const [progress, setProgress] = useState([0] * chatHistory.length)

  useEffect(() => {
    if (suppress == 0) {
        return;
    }
    console.log("trigger: " + trigger + new Date().toLocaleTimeString())
    const wordIndices = getWordIndices(chatHistory)
    const progress= getProgress(chatHistory)
  }, [trigger]);

  return(
    <div className="space-y-2">
    {chatHistory.map((entry, index) => (
        <div key={index} className="p-2 border rounded">
        <strong>Role {entry.role}:</strong>
        <SentenceHighlight sentence={entry.message} indices={wordIndices[index] || []} />
        {trigger && <progress max="100" value="80"></progress>}
        </div>
    ))}
    </div>
  )
}

export default function DialogLabel({ chatHistory, trigger, suppress }) {
  const [label, setLabel] = useState("");

  useEffect(() => {
    if (suppress == 0) {
        return;
    }
    console.log("trigger: " + trigger + new Date().toLocaleTimeString())
    const result = getDialogLabel(chatHistory);
    setLabel(result);
  }, [trigger]);

  return (
    <div className="mb-4">
      <p className="text-lg">
        <strong>Dialogue Label:</strong> {label}
      </p>
    </div>
  );
}
