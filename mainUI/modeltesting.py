from gpt4all import GPT4All

model4 = GPT4All("C:\\Users\\comac\\AppData\\Local\\nomic.ai\\GPT4All\\wizardlm-13b-v1.2.Q4_0.gguf", allow_download=False)
# model5 = GPT4All("orca-mini-3b-gguf2-q4_0.gguf", allow_download=False)

# output = model3.generate("The capital of France is ", max_tokens=20)

# print(output)

sysprompt = "use one word to classify depression severity of the input text: no, mild, moderate, moderately severe, or severe risk of depression"

def test_model3():         
    with model4.chat_session(sysprompt):
        response1 = model4.generate("hello i am not depressed at all can you tell or not") 
        return model4.current_chat_session.pop()['content'].lower()
print(test_model3())
# system_template = 'use just one word to classify the input text according to PHQ level of depression severity with only one of these words: no, mild, moderate, moderately severe, severe'
# prompt_template = '[INST] %1 [/INST]'
# prompts = ['Im going to kill myself', 'now name 3 fruits', 'what were the 3 colors in your earlier response?']
# first_input = system_template + prompt_template.format(prompts[0])
# response = model4.generate(first_input, temp=0)
# print(response)

# with model5.chat_session(sysprompt, template):
#     response1 = model5.generate(prompt="I'm going to kill myself")
#     print(model5.current_chat_session.pop()['content'])
    
# print(output)

# print(model4.generate("i want to kill myself"))
