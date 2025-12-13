import React, { useState } from 'react';
import styles from './Feedback.module.css';

interface FeedbackProps {
  moduleId: string;
  title: string;
}

const Feedback: React.FC<FeedbackProps> = ({ moduleId, title }) => {
  const [feedback, setFeedback] = useState({
    rating: 0,
    comment: '',
    difficulty: 'intermediate'
  });
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Save feedback to localStorage
    const feedbackData = {
      ...feedback,
      moduleId,
      timestamp: new Date().toISOString(),
      title
    };

    localStorage.setItem(`feedback-${moduleId}`, JSON.stringify(feedbackData));
    setSubmitted(true);

    // Show confirmation message
    alert('Thank you for your feedback!');
  };

  const handleRating = (rating: number) => {
    setFeedback({ ...feedback, rating });
  };

  if (submitted) {
    return (
      <div className={styles.feedbackContainer}>
        <div className={styles.feedbackConfirmation}>
          <h3>Thank You!</h3>
          <p>Your feedback has been recorded.</p>
          <button
            className={styles.resetBtn}
            onClick={() => setSubmitted(false)}
          >
            Submit Another Feedback
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.feedbackContainer}>
      <h3>Assessment Feedback: {title}</h3>

      <form onSubmit={handleSubmit} className={styles.feedbackForm}>
        <div className={styles.ratingSection}>
          <label>How would you rate this assessment?</label>
          <div className={styles.ratingStars}>
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                key={star}
                type="button"
                className={`${styles.star} ${feedback.rating >= star ? styles.filled : ''}`}
                onClick={() => handleRating(star)}
              >
                ★
              </button>
            ))}
          </div>
        </div>

        <div className={styles.difficultySection}>
          <label>How difficult was this assessment?</label>
          <select
            value={feedback.difficulty}
            onChange={(e) => setFeedback({ ...feedback, difficulty: e.target.value })}
            className={styles.select}
          >
            <option value="beginner">Beginner</option>
            <option value="intermediate">Intermediate</option>
            <option value="advanced">Advanced</option>
            <option value="challenging">Challenging</option>
          </select>
        </div>

        <div className={styles.commentSection}>
          <label>Additional Comments:</label>
          <textarea
            value={feedback.comment}
            onChange={(e) => setFeedback({ ...feedback, comment: e.target.value })}
            className={styles.textarea}
            placeholder="Share your thoughts about this assessment..."
            rows={4}
          />
        </div>

        <button type="submit" className={styles.submitBtn}>
          Submit Feedback
        </button>
      </form>
    </div>
  );
};

export default Feedback;