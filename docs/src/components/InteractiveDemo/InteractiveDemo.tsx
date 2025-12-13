import React, { useState } from 'react';
import clsx from 'clsx';
import styles from './InteractiveDemo.module.css';

interface InteractiveDemoProps {
  title: string;
  description?: string;
  children?: React.ReactNode;
  defaultCode?: string;
  language?: string;
}

const InteractiveDemo: React.FC<InteractiveDemoProps> = ({
  title,
  description,
  children,
  defaultCode = '',
  language = 'python'
}) => {
  const [code, setCode] = useState(defaultCode);
  const [output, setOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);

  const handleRunCode = () => {
    setIsRunning(true);
    // In a real implementation, this would connect to a backend service
    // to execute the code safely. For now, we'll simulate execution.
    setTimeout(() => {
      setOutput(`Executed ${language} code successfully!\nOutput would appear here in a real implementation.`);
      setIsRunning(false);
    }, 500);
  };

  const handleResetCode = () => {
    setCode(defaultCode);
    setOutput('');
  };

  return (
    <div className={clsx('margin--md', styles.interactiveDemoContainer)}>
      <div className={styles.interactiveDemoHeader}>
        <h3>{title}</h3>
        {description && <p>{description}</p>}
      </div>

      <div className={styles.codeEditor}>
        <div className={styles.editorHeader}>
          <span className={styles.languageLabel}>{language}</span>
          <div className={styles.editorActions}>
            <button
              className={clsx('button button--sm button--primary', styles.runButton)}
              onClick={handleRunCode}
              disabled={isRunning}
            >
              {isRunning ? 'Running...' : 'Run Code'}
            </button>
            <button
              className={clsx('button button--sm button--secondary', styles.resetButton)}
              onClick={handleResetCode}
            >
              Reset
            </button>
          </div>
        </div>
        <textarea
          className={clsx('form-control', styles.codeTextarea)}
          value={code}
          onChange={(e) => setCode(e.target.value)}
          spellCheck="false"
          rows={10}
        />
      </div>

      {output && (
        <div className={styles.outputPanel}>
          <div className={styles.outputHeader}>
            <span>Output</span>
          </div>
          <pre className={styles.outputContent}>
            <code>{output}</code>
          </pre>
        </div>
      )}

      {children && <div className={styles.demoContent}>{children}</div>}
    </div>
  );
};

export default InteractiveDemo;