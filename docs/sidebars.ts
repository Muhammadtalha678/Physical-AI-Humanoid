import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: ['intro'],
    },
    {
      type: 'category',
      label: 'Week 1-2: Introduction to Physical AI',
      items: [
        'week-01-02-introduction/foundations-of-physical-ai',
        'week-01-02-introduction/from-digital-ai-to-physical-systems',
        'week-01-02-introduction/overview-humanoid-robotics',
      ],
    },
    {
      type: 'category',
      label: 'Week 3-5: ROS 2 Fundamentals',
      items: [
        'week-03-05-ros2/ros2-architecture-concepts',
        'week-03-05-ros2/nodes-topics-services-actions',
        'week-03-05-ros2/building-ros2-packages-python',
        'week-03-05-ros2/launch-files-parameter-management',
      ],
    },
    {
      type: 'category',
      label: 'Week 6-7: Robot Simulation with Gazebo',
      items: [
        'week-06-07-simulation/gazebo-simulation-setup',
        'week-03-05-ros2/urdf-sdf-robot-descriptions',
        'week-03-05-ros2/physics-simulation-sensors',
        'week-03-05-ros2/unity-visualization',
      ],
    },
    {
      type: 'category',
      label: 'Week 8-10: NVIDIA Isaac Platform',
      items: [
        'week-08-10-isaac/introduction-to-isaac-platform',
        'week-08-10-isaac/ai-powered-perception-manipulation',
        'week-08-10-isaac/reinforcement-learning-sim-to-real',
      ],
    },
    {
      type: 'category',
      label: 'Week 11-12: Humanoid Robot Development',
      items: [
        'week-11-12-humanoid/humanoid-kinematics-dynamics',
        'week-11-12-humanoid/bipedal-locomotion-human-robot-interaction',
      ],
    },
    {
      type: 'category',
      label: 'Week 13: Conversational Robotics',
      items: [
        'week-13-conversational/conversational-ai-capstone',
      ],
    },
    {
      type: 'category',
      label: 'Assessment Projects',
      items: [
        'projects/ros2-package-dev',
        'projects/gazebo-simulation-impl',
        'projects/isaac-perception-pipeline',
        'projects/capstone-autonomous-humanoid',
      ],
    },
    {
      type: 'category',
      label: 'Information',
      items: [
        'accessibility',
      ],
    },
  ],
};

export default sidebars;
