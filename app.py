import random
import time
import string

def dummy_llm(input_text):
    """
    Simulates an LLM by generating gibberish text based on input length.
    Adds fake "thinking" delay for realism.
    """
    
    # Generate output length based on input length
    output_length = len(input_text) * random.randint(2, 4)
    
    # Generate random words of random lengths
    words = []
    current_length = 0
    
    while current_length < output_length:
        word_length = random.randint(3, 12)
        word = ''.join(random.choice(string.ascii_lowercase) for _ in range(word_length))
        words.append(word)
        current_length += word_length + 1  # +1 for space
    
    # Add some punctuation
    punctuation = ['.', '!', '?', ',']
    for i in range(len(words) // 4):  # Add punctuation every ~4 words
        pos = random.randint(0, len(words) - 1)
        words[pos] = words[pos] + random.choice(punctuation)
    
    return ' '.join(words)

def main():
    print("Welcome to the Dummy LLM Terminal Interface!")
    print("Enter your text (or 'quit' to exit):")
    
    while True:
        try:
            # Get input from user
            user_input = input("\nYou: ").strip()
            
            # Check for exit condition
            if user_input.lower() == 'quit':
                print("\nGoodbye!")
                break
            
            if not user_input:
                print("Please enter some text!")
                continue
            
            # Show "thinking" animation
            print("Processing", end="")
            for _ in range(3):
                time.sleep(0.5)
                print(".", end="", flush=True)
            print("\n")
            
            # Get response from dummy LLM
            response = dummy_llm(user_input)
            
            # Print response
            print("LLM:", response)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()