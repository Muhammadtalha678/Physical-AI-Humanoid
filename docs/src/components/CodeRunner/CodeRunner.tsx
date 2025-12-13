import React, { useState } from 'react';
import clsx from 'clsx';
import styles from './CodeRunner.module.css';

interface CodeRunnerProps {
  title?: string;
  description?: string;
  initialCode?: string;
  language?: string;
  showLineNumbers?: boolean;
}

const CodeRunner: React.FC<CodeRunnerProps> = ({
  title = 'Code Runner',
  description,
  initialCode = '# Write your code here\nprint("Hello, Physical AI!")',
  language = 'python',
  showLineNumbers = true
}) => {
  const [code, setCode] = useState(initialCode);
  const [output, setOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [executionTime, setExecutionTime] = useState<number | null>(null);

  const handleRunCode = () => {
    setIsRunning(true);
    setOutput('');
    setExecutionTime(null);

    // In a real implementation, this would connect to a backend service
    // to execute the code safely. For now, we'll simulate execution.
    setTimeout(() => {
      const startTime = performance.now();
      // Simulate execution
      setOutput(`Executed ${language} code successfully!\n\nYour code would run in a secure sandbox environment.\n\nSimulated output based on your code.`);
      const endTime = performance.now();
      setExecutionTime(endTime - startTime);
      setIsRunning(false);
    }, 800);
  };

  const handleResetCode = () => {
    setCode(initialCode);
    setOutput('');
    setExecutionTime(null);
  };

  const handleClearOutput = () => {
    setOutput('');
    setExecutionTime(null);
  };

  return (
    <div className={clsx('margin--md', styles.codeRunnerContainer)}>
      <div className={styles.codeRunnerHeader}>
        {title && <h3>{title}</h3>}
        {description && <p className={styles.description}>{description}</p>}
      </div>

      <div className={styles.editorSection}>
        <div className={styles.editorHeader}>
          <div className={styles.languageInfo}>
            <span className={clsx('badge badge--secondary', styles.languageBadge)}>
              {language}
            </span>
          </div>
          <div className={styles.editorControls}>
            <button
              className={clsx('button button--sm button--primary', styles.runButton)}
              onClick={handleRunCode}
              disabled={isRunning}
            >
              {isRunning ? (
                <>
                  <span className={styles.spinner}></span> Running...
                </>
              ) : (
                '▶ Run'
              )}
            </button>
            <button
              className={clsx('button button--sm button--secondary', styles.resetButton)}
              onClick={handleResetCode}
              title="Reset to initial code"
            >
              ♺ Reset
            </button>
            <button
              className={clsx('button button--sm button--outline button--secondary', styles.clearButton)}
              onClick={handleClearOutput}
              title="Clear output"
            >
              Clear
            </button>
          </div>
        </div>

        <div className={styles.codeEditor}>
          <textarea
            className={clsx('form-control', styles.codeTextarea)}
            value={code}
            onChange={(e) => setCode(e.target.value)}
            spellCheck="false"
            rows={12}
          />
        </div>
      </div>

      {(output || executionTime) && (
        <div className={styles.outputSection}>
          <div className={styles.outputHeader}>
            <span className="badge badge--success">Output</span>
            {executionTime !== null && (
              <span className={styles.executionTime}>
                Execution time: {executionTime.toFixed(2)}ms
              </span>
            )}
          </div>
          <div className={styles.outputContent}>
            <pre>
              <code>{output}</code>
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};

export default CodeRunner;