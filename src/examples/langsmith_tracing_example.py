import openai
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai

_ = load_dotenv()

# Auto-trace LLM calls in-context
client = wrap_openai(openai.Client())


@traceable  # Auto-trace this function
def pipeline(user_input: str):
    result = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}], model="gpt-3.5-turbo"
    )
    return result.choices[0].message.content


print(pipeline("Hello, world!"))


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Please respond to the user's request only based on the given context.",
        ),
        ("user", "Question: {question}\nContext: {context}"),
    ]
)
model = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

question = "Can you summarize the following text?"
context = """The **MacBook** is a line of Macintosh laptop computers designed and marketed by Apple Inc. It runs on Apple's macOS operating system and is known for its sleek design, high performance, and user-friendly interface. The MacBook family includes the **MacBook Air** and **MacBook Pro** models, each catering to different user needs.

### Key Features:
- **Design**: MacBooks are renowned for their minimalist, elegant design, featuring a unibody aluminum construction. They are lightweight and portable, making them ideal for on-the-go use.
- **Display**: They come with high-resolution Retina displays, offering vibrant colors and sharp details, perfect for creative professionals and media consumption.
- **Performance**: Equipped with Apple's custom silicon chips (like the M1, M2, and the latest M3 series), MacBooks deliver powerful performance with excellent energy efficiency.
- **Battery Life**: Known for their long battery life, MacBooks can last up to 20 hours on a single charge, depending on the model and usage.
- **Keyboard and Trackpad**: They feature a backlit Magic Keyboard and a large Force Touch trackpad, providing a comfortable and responsive typing and navigation experience.
- **Connectivity**: Modern MacBooks include Thunderbolt/USB 4 ports, Wi-Fi 6, and Bluetooth 5.0, ensuring fast and reliable connectivity.
- **Security**: They come with advanced security features like Touch ID and the Apple T2 Security Chip, ensuring your data is safe and secure.

Whether you need a lightweight laptop for everyday tasks or a powerful machine for professional work, there's a MacBook model to suit your needs¹²³.

Is there a specific MacBook model you're interested in, or do you have any other questions about them?
"""
result = chain.invoke({"question": question, "context": context})
print(result)
