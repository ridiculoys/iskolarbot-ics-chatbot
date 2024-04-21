# A Dynamic Query Framework for Research Accessibility using OpenAI and Langchain
Authors: [Louise Gabrielle L. Talip](https://github.com/ridiculoys) and Reginald Neil C. Recario

Keywords: chatbot, research assistant, openai, langchain, chainlit, pdf reader


# Development Set-up

## Prerequisites
* Python 3.10 or above
* Libraries specified in the `requirements.txt` file
* VS Code or any IDE that can open Python files
* Windows 10 OS - can run on Windows or WSL shell

## Instructions
1. Make sure to prepare the .env file (See .env.example)
2. To run the application, run these on the terminal 

    WSL/Ubuntu:
    ```bash
      python -m venv venv
      source venv/bin/Activate
      pip install -r requirements.txt
      chainlit run app.py
    ```

    Windows:
    ```bash
      python -m venv venv
      venv/Scripts/Activate
      pip install -r requirements.txt
      chainlit run app.py
    ```

## For setting up and changing the datasets:

0. Setup `.env`, virtual environment, and requirements (see above).
1. Before running the app, make sure to put the documents in the `data` folder.
2. Update the variables in `app_setup_index.py`
3. Run `python app_setup_index.py`
4. Update the variables in `app_setup_docs.py`
5. Run `python app_setup_docs.py`
6. Update the `init_search_content` function in `app.py`. Update the `value` for each action to reflect the index name of the respective pinecone index of the datasets.
7. Run `chainlit run app.py`


# This application is also hosted [here](https://test-ics-chatbot.onrender.com/)