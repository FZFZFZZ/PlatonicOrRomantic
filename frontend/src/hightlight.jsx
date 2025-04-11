import React, { useState, useEffect } from 'react';

export const SentenceHighlight = ({ sentence, indices, trigger, suppress }) => {
  const words = sentence.split(' ');
  const [isHighlight, setIsHighlight] = useState(0)

  useEffect(() => {
    if (suppress == 0) {
      setIsHighlight(0)
      return;
    }
    setIsHighlight(1)
  }, [trigger]);

  return (
    <div>
      {words.map((word, index) => {
        // Check if the current word index is in the indices array
        const isHighlighted = (isHighlight == 1) && indices.includes(index);

        return (
          <span
            key={index}
            style={{
              backgroundColor: isHighlighted ? 'yellow' : 'transparent', // Apply highlight only if the word index matches
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
