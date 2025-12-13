Create a Retrieval-Augmented Generation (RAG) backend system with fastapi, Qdrant vector db and OpenAIAgent sdk. Flow Two apis create using fast api one is for embeddings, this embedding api take the path of the directory where .md extension files are presented. These files are presented inside the docasarus project having directory frontend/docs/.md files,.md files,subfolders having .md files Second is for search query when user ask question the api hit of query this query than go to OpnAiAgentSDK than this agent see the query is related to greetings than the agent responseke normal greeeting with "How can I help you with Physical AI & Humanoid Book",something like that in greeting, If the query is not related to greeting than call the agent tool with query than the tool search inside the qdrant vector db if the query related to book question than the agent  response with the same book response, if nothing found from book than the agent response like nothing found related to this query found, or if user ask any impropiate query than agent reply with "Be in a manner like message", This agents responses return back with the second api in response to the user. The qdrant_api ,qdrant_url,qdrant_ection_name,embedding_model,google_api_key,openrouteer_api,openrouter_baseurl,openrouter_model_name these all I provide.


During
 /sp.specify: Query the RAG backend for relevant documentation, best practices, and patterns related to the feature being specified

  2. During /sp.plan: Retrieve architectural patterns, technology recommendations, and implementation strategies from the knowledge base

  3. During /sp.tasks: Access task templates, common implementation patterns, and testing strategies

  4. During /sp.implement: Get code examples, API documentation, and best practices for the actual implementation



    
  Before anything else, we should outline the textbook—its structure, sections, chapters—and prepare the
  Docusaurus project, including layout and design.
  
  ## Background information:
  \n
  The textbook supports a 13-week "Physical AI & Humanoid Robotics" training program aimed at working
  professionals,Intended readers:
  \n
  industry engineers who already know Python,
  The book will be published using Docusaurus and deployed via GitHub Pages,
  The curriculum is hardware-neutral and uses Python,
  ROS 2, and Isaac Sim,
  Course structure:

  ## Weeks 1-2: Introduction to Physical AI
  \n
  Foundations of Physical AI and embodied intelligence
  From digital AI to robots that understand physical laws
  Overview of humanoid robotics landscape
  Sensor systems: LIDAR, cameras, IMUs, force/torque sensors
  \n
  ## Weeks 3-5: ROS 2 Fundamentals
  \n
  ROS 2 architecture and core concepts
  Nodes, topics, services, and actions
  Building ROS 2 packages with Python
  Launch files and parameter management
  \n
  ## Weeks 6-7: Robot Simulation with Gazebo
  \n
  Gazebo simulation environment setup
  URDF and SDF robot description formats
  Physics simulation and sensor simulation
  Introduction to Unity for robot visualization
  \n
  ## Weeks 8-10: NVIDIA Isaac Platform
  \n
  NVIDIA Isaac SDK and Isaac Sim
  AI-powered perception and manipulation
  Reinforcement learning for robot control
  Sim-to-real transfer techniques
  \n
  ## Weeks 11-12: Humanoid Robot Development
  \n
  Humanoid robot kinematics and dynamics
  Bipedal locomotion and balance control
  Manipulation and grasping with humanoid hands
  Natural human-robot interaction design
  \n
  ## Weeks 13: Conversational Robotics
  \n
  Integrating GPT models for conversational AI in robots
  Speech recognition and natural language understanding
  Multi-modal interaction: speech, gesture, vision
  \n
  ## Assessments
  \n
  ROS 2 package development project
  Gazebo simulation implementation
  Isaac-based perception pipeline
  Capstone: Simulated humanoid robot with conversational AI
  \n
  ## Final capstone
  \n
  Build an autonomous humanoid pipeline (speech → planning → navigation → perception →
  manipulation),Learners can choose from three platform setups: Digital Twin workstation, Physical AI Edge Kit, or a
  cloud-native environment.
  \n
  ## Notice
  \n
  I already connected the Context7 MCP server with my project to access Docusaurus
  documentation to create project with npx and select typescript with classic
  . Must create project name docs using this command fetch from context7 mcp server
  and refine the book master plan spec with Docusaurus-specific clarifications.
  ## 002-physical-ai-textbook