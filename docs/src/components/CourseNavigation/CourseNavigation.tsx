import React from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import clsx from 'clsx';
import styles from './CourseNavigation.module.css';

interface CourseNavigationProps {
  currentWeek: string;
  weeks: {
    id: string;
    title: string;
    path: string;
    completed?: boolean;
  }[];
}

const CourseNavigation: React.FC<CourseNavigationProps> = ({
  currentWeek,
  weeks
}) => {
  return (
    <nav className={styles.courseNavigation}>
      <div className={styles.navigationHeader}>
        <h3 className={styles.navigationTitle}>Course Navigation</h3>
      </div>

      <ul className={styles.weekList}>
        {weeks.map((week) => (
          <li
            key={week.id}
            className={clsx(
              styles.weekItem,
              week.id === currentWeek && styles.weekItemActive,
              week.completed && styles.weekItemCompleted
            )}
          >
            <Link
              to={useBaseUrl(week.path)}
              className={clsx(
                styles.weekLink,
                week.id === currentWeek && styles.weekLinkActive
              )}
            >
              <span className={styles.weekId}>{week.id}</span>
              <span className={styles.weekTitle}>{week.title}</span>
              {week.completed && (
                <span className={styles.completionBadge}>✓</span>
              )}
            </Link>
          </li>
        ))}
      </ul>
    </nav>
  );
};

export default CourseNavigation;