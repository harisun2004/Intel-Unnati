from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig
from intel_extension_for_transformers.transformers import MixedPrecisionConfig

# Setup chatbot configuration
config = PipelineConfig(optimization_config=MixedPrecisionConfig())
chatbot = build_chatbot(config)

# Interaction loop
while True:
    # Prompt user for input
    user_input = input("Ask a question (type 'end' to exit): ")
    
    # Check if user wants to end the conversation
    if user_input.strip().lower() == 'end':
        print("Ending conversation...")
        break
    
    # Get response from the chatbot
    response = chatbot.predict(query=user_input)
    print("Chatbot:", response)
