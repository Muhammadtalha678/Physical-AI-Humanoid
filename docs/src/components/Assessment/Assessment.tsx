import React, { useState } from 'react';
import clsx from 'clsx';
import styles from './Assessment.module.css';

interface AssessmentOption {
  id: string;
  text: string;
}

interface AssessmentProps {
  question: string;
  options: AssessmentOption[];
  correctAnswerId: string;
  explanation?: string;
  type?: 'multiple-choice' | 'single-choice';
  onSubmit?: (isCorrect: boolean, selectedOption: string) => void;
}

const Assessment: React.FC<AssessmentProps> = ({
  question,
  options,
  correctAnswerId,
  explanation,
  type = 'single-choice',
  onSubmit
}) => {
  const [selectedOption, setSelectedOption] = useState<string | null>(null);
  const [submitted, setSubmitted] = useState(false);
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null);

  const handleOptionChange = (optionId: string) => {
    if (!submitted) {
      setSelectedOption(optionId);
    }
  };

  const handleSubmit = () => {
    if (selectedOption === null) return;

    const correct = selectedOption === correctAnswerId;
    setIsCorrect(correct);
    setSubmitted(true);

    if (onSubmit) {
      onSubmit(correct, selectedOption);
    }
  };

  const handleReset = () => {
    setSelectedOption(null);
    setSubmitted(false);
    setIsCorrect(null);
  };

  return (
    <div className={clsx('margin--md', styles.assessmentContainer)}>
      <div className={styles.assessmentHeader}>
        <h4 className={styles.question}>{question}</h4>
      </div>

      <div className={styles.optionsContainer}>
        {options.map((option) => (
          <div
            key={option.id}
            className={clsx(
              styles.option,
              selectedOption === option.id && styles.optionSelected,
              submitted && option.id === correctAnswerId && styles.optionCorrect,
              submitted && selectedOption === option.id && selectedOption !== correctAnswerId && styles.optionIncorrect
            )}
            onClick={() => handleOptionChange(option.id)}
          >
            <input
              type={type === 'single-choice' ? 'radio' : 'checkbox'}
              id={`option-${option.id}`}
              name="assessment-options"
              checked={selectedOption === option.id}
              onChange={() => {}}
              disabled={submitted}
              className={styles.optionInput}
            />
            <label
              htmlFor={`option-${option.id}`}
              className={clsx('padding--sm', styles.optionLabel)}
            >
              {option.text}
            </label>
          </div>
        ))}
      </div>

      {!submitted ? (
        <div className={styles.submitSection}>
          <button
            className={clsx('button button--primary', styles.submitButton)}
            onClick={handleSubmit}
            disabled={selectedOption === null}
          >
            Submit Answer
          </button>
        </div>
      ) : (
        <div className={styles.resultSection}>
          <div className={clsx(
            'alert',
            isCorrect ? 'alert--success' : 'alert--error',
            styles.resultMessage
          )}>
            <div className={styles.resultContent}>
              <span className={styles.resultIcon}>
                {isCorrect ? '✓' : '✗'}
              </span>
              <span className={styles.resultText}>
                {isCorrect ? 'Correct!' : 'Incorrect.'}
              </span>
            </div>
          </div>

          {explanation && (
            <div className={clsx('alert alert--info', styles.explanation)}>
              <h5>Explanation:</h5>
              <p>{explanation}</p>
            </div>
          )}

          <div className={styles.resetSection}>
            <button
              className={clsx('button button--secondary', styles.resetButton)}
              onClick={handleReset}
            >
              Try Again
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Assessment;