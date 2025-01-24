# 🖥️ Compound AI System using LLMs 🖥️

## Introduction

As part of the [Berkley LLM Agent Hackathon 2024](https://rdi.berkeley.edu/llm-agents-hackathon/), this multi-agent solution was developed to solve real world problems such as: `financial inclusion` and `financial information asymmetry`, `lack of computer science/programming teachers or overburdened teachers` especially in developing countries such as *India*, help with `wikipedia research`, stay `abreast with contemporary news` and a personal `Question Answering` system.
This showcases the potential of making education accessible, and aid vital information dissemination.

In essence, this *compound AI system* showcases the benefits of creating such a system, by orchestrating several agents to solve complex tasks and accessing it all from a single screen, and all while having a conversation with it.

## System Features

This system has the following features:

1. Search top 5 news from hackernews and return a summary of the articles
2. Search top news from the web
3. Act as a personal financial planner
4. Return equity, analyst recommendations, and company news for publicly listed
companies in USA.
5. Search Wikipedia.
6. C, C++, Rust, Python Programming tutor.
7. Ask me anything(AMA).

A more detailed description is [here](#brief-description-of-the-system-features).

### Safety Features

The system has a `content moderation` in place, which checks for harmful content being sent to it and asked to generate and **REFUSES** to answer the query.
It also has the ability to detect `prompt leakage/attack` and refuses to divulge any compromising information such as the *API KEY, System Prompt,...* .

## Usage
<span style="color:red;">**Follow the instructions precisely.**</span>

1. Navigate to `berkley_hack` dir, this will be default name of the dir, if you clone this repo
2. Create a `.env` file and input your `OPENAI_API_KEY=`
3. Install [UV](https://docs.astral.sh/uv/getting-started/installation/), if unavailable
4. Execute `uv run src/app.py`
5. Navigate to `http://127.0.0.1:7860`
6. Happy Chatting! 😉

## Brief description of the system features

### Search top 5 news from hackernews and return a summary of the articles

The agent searches through [hackernews](https://thehackernews.com/) for the latest articles on a topic of choice and retrieves the *most recent and most relevant* top 5 articles from it and summarises each article and returns the article headline, URL and the summary.

### Search top news from the web

This is used for searching top news, in English, across the internet and returns top 5 news.

### Personal Financial Planner

This acts as a financial planner and tries to guide each individual to have a financial plan in place for a better financial security and future. It can answer many personal finance questions and give recommendations based on the information you provide.

<span style="color:red;">**Disclaimer:**</span>
All personal finance recommendations from the system are solely for demonstration purposes only. It's solely up to the user's discretion, whether to use the recommendations provided to him/her.

### Equity Researcher

This feature leverages publicly available information about a publicly listed company in USA. It can retrieve the current equity price, analyst recommendations for further research, most recent company news about the said company and even *compare companies tête à tête*. It's an useful tool for researching about a publicly listed company before investing in it.
The detailed research is also available as separate markdown files for the user to leverage.

### Wikipedia Rearcher

It leverages, **Wikipedia(Thank you Wikipedia 🙏)**, to research, simplify and summarise a particular subject, for which there's an existing Wikipedia page. If there's no Wikipedia page, then the system leverages public information from other websites and attempts to give a solution.

### Personal Programming Tutor

As the name suggests, this feature helps anyone interested in learning a programming language by providing personalised lesson plans as per the progress of the student. Currently, the tutor is confined to some of the high resource languages but it can also be used for low resource languages with further modifications to the system.

### Ask me anything(AMA)

This is self-explanatory, and it aims to answer any question that the user might have, all under the ambit of the internet of course 😉.

# Resources

- [Video Presentation](https://www.loom.com/share/9d721891941c49f095fd5fec8cb087bb?sid=35c80e74-b3fe-4877-9432-053a3abe081d)
- [Slides](https://docs.google.com/presentation/d/1jrONLX6S9hsc4My43XxR44jb0WzpISy5SM5OwJcB4y0/edit?usp=sharing)
