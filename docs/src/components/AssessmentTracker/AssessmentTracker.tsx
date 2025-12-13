import React, { useState, useEffect } from 'react';
import styles from './AssessmentTracker.module.css';

interface AssessmentTrackerProps {
  moduleId: string;
  title: string;
}

const AssessmentTracker: React.FC<AssessmentTrackerProps> = ({ moduleId, title }) => {
  const [completed, setCompleted] = useState(false);
  const [progress, setProgress] = useState(0);

  // Load progress from localStorage
  useEffect(() => {
    const savedProgress = localStorage.getItem(`assessment-${moduleId}`);
    if (savedProgress) {
      const parsed = JSON.parse(savedProgress);
      setCompleted(parsed.completed);
      setProgress(parsed.progress);
    }
  }, [moduleId]);

  // Save progress to localStorage
  const saveProgress = (completedStatus: boolean, progressValue: number) => {
    const data = {
      completed: completedStatus,
      progress: progressValue,
      timestamp: new Date().toISOString()
    };
    localStorage.setItem(`assessment-${moduleId}`, JSON.stringify(data));
    setCompleted(completedStatus);
    setProgress(progressValue);
  };

  const markAsComplete = () => {
    saveProgress(true, 100);
  };

  const markAsIncomplete = () => {
    saveProgress(false, 0);
  };

  return (
    <div className={styles.trackerContainer}>
      <div className={styles.trackerHeader}>
        <h3>Assessment Progress: {title}</h3>
        <div className={styles.progressInfo}>
          Progress: {progress}%
        </div>
      </div>

      <div className={styles.progressBar}>
        <div
          className={styles.progressFill}
          style={{ width: `${progress}%` }}
        ></div>
      </div>

      <div className={styles.trackerActions}>
        {!completed ? (
          <button
            className={styles.completeBtn}
            onClick={markAsComplete}
          >
            Mark as Complete
          </button>
        ) : (
          <button
            className={styles.incompleteBtn}
            onClick={markAsIncomplete}
          >
            Mark as Incomplete
          </button>
        )}
      </div>

      <div className={styles.status}>
        Status: {completed ? 'Completed' : 'In Progress'}
      </div>
    </div>
  );
};

export default AssessmentTracker;