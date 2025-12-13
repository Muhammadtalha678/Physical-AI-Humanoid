import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg?: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Physical AI & Embodied Intelligence',
    description: (
      <>
        Learn the fundamentals of Physical AI - the integration of perception, reasoning,
        and action in embodied systems. Explore how robots can understand and interact
        with the physical world using advanced AI techniques.
      </>
    ),
  },
  {
    title: 'ROS 2 & Robotics Development',
    description: (
      <>
        Master ROS 2 (Robot Operating System 2) for building sophisticated robotic
        applications. Learn about nodes, topics, services, and actions while developing
        real-world robotics projects.
      </>
    ),
  },
  {
    title: 'Humanoid Robotics & AI',
    description: (
      <>
        Dive into humanoid robotics development, including locomotion, manipulation,
        and human-robot interaction. Learn to build and program bipedal robots with
        conversational AI capabilities.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
