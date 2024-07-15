# Intel-Unnati-GenAI-Chatbot

## Introduction to GenAI and Simple LLM Inference on CPU and fine-tuning of LLM Model to create a Custom Chatbot

### Description:

The rapid advancements in Generative Artificial Intelligence (GenAI) have paved the way for sophisticated conversational agents known as chatbots. These chatbots, powered by Large Language Models (LLMs), have shown remarkable capabilities in understanding and generating human-like text. However, achieving optimal performance and customization often requires fine-tuning these models on specific datasets and hardware optimizations.

This project focuses on enhancing chatbot capabilities through the following key objectives:

Inference on CPU: Understanding and implementing Large Language Model inference on CPU platforms, leveraging Intel's optimizations for efficient processing.

Fine-tuning LLMs: Exploring the concept of fine-tuning, a crucial technique in machine learning where pre-trained models are adapted to specific tasks or datasets. Fine-tuning will be performed using Intel tools to maximize performance and efficiency.

Custom Chatbot Development: Developing a custom chatbot by integrating the fine-tuned LLM, allowing it to respond intelligently to user queries and maintain engaging conversations.

Intel Optimization: Utilizing Intel's optimizations and tools to enhance the performance of the chatbot, ensuring efficient computation and scalability.

### Objectives:

Implement LLM Inference: Demonstrate the ability to perform Large Language Model inference on CPU, optimizing performance using Intel tools.

Fine-tune LLM: Fine-tune a pre-trained LLM on a dataset relevant to chatbot applications, ensuring the model learns task-specific nuances effectively.

Develop Custom Chatbot: Integrate the fine-tuned LLM into a chatbot application, allowing it to provide accurate and contextually relevant responses.

Utilize Intel Optimizations: Apply Intel's optimizations to improve the efficiency and speed of the chatbot's inference and training processes.

### Expected Outcome:

Participants will gain hands-on experience in working with cutting-edge AI technologies, understanding the intricacies of LLMs, and leveraging Intel optimizations for enhanced performance. By the end of this project, participants will have developed and deployed a custom chatbot capable of intelligent conversations, showcasing the practical application of GenAI in real-world scenarios.

This problem statement provides a comprehensive overview of the project's objectives, challenges, and expected outcomes, focusing on enhancing chatbot capabilities through LLM fine-tuning and Intel optimizations. Adjust the specifics based on your project's scope and goals for clarity and relevance.

### Major Challenges:

1. Pre-trained Language Models can have large file sizes, which may require significant storage space
and memory to load and run.
2. Learn LLM inference on CPU.
3. Understanding the concept of fine-tuning and its importance in customizing LLMs.
4. Create a Custom Chatbot with fine-tuned pre-trained Large Language Models (LLMs) using Intel AI
Tools.

### Outcomes:

1. Participants will gain a foundational understanding of Generative AI and its applications.
2. Participants will be able to perform simple LLM inference on a CPU and understand the process of
fine-tuning LLMs for custom applications.
3. Participants will have to create a 5-page report on Problem, Technical Approach and Results. 

### Creation of environment

1. Creation of environment and activating it
```bash
python -m venv intel
source intel/bin/activate
```
2. Installing intel-extention-for-transformers
```bash
pip install intel-extension-for-transformers
```
3. Cloning Github repository for intel extensions 
```bash
git clone https://github.com/intel/intel-extension-for-transformers.git
cd ./intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/
```
4. Install Requirements
```bash
pip install -r requirements_cpu.txt
pip install -r requirements.txt
cd /../../
```
5. Logging into huggingface and providing access tokens
```bash
huggingface-cli login
```
6. Running the project
```bash
python finetune.py
python chatbot.py
```
### Output


### Dataset
We use the Alpaca dataset [https://github.com/tatsu-lab/stanford_alpaca] from Stanford University as the general domain dataset to fine-tune the model. This dataset is provided in the form of a JSON file, alpaca_data.json. In Alpaca, researchers have manually crafted 175 seed tasks to guide text-davinci-003 in generating 52K instruction data for diverse tasks.
