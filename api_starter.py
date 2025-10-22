#!/usr/bin/env python3

"""
https://aistudio.google.com/ "Get API key"
https://aistudio.google.com/apikey
Put key in .env file with format:
GEMINI_API_KEY="the key"
Use OpenAIServerModel due to API compatibility
[Select model](https://ai.google.dev/gemini-api/docs/models)
"""

#############################################
# Environment loading
#############################################
import pandas as pd
import numpy as np
import dotenv
import os
import time
import json
g_dotenv_loaded = False
def getenv(variable: str) -> str:
    global g_dotenv_loaded
    if not g_dotenv_loaded:
        g_dotenv_loaded = True
        dotenv.load_dotenv()
    value = os.getenv(variable)
    return value

api_key = getenv("GEMINI_API_KEY")


if not api_key:
    raise Exception("GEMINI_API_KEY needs to be set in .env.")

#############################################
# Model connection
#############################################
from smolagents import OpenAIServerModel

#model_id="gemini-2.0-flash"
#model_id="gemini-2.0-flash-lite"
model_id="gemini-2.5-flash"
model = OpenAIServerModel(model_id=model_id,
                          api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                          api_key=api_key,
                          )

ABC = ["a","b","c","d", "e","f","g","h","i","j","k"]

############################################
# puting the data into groups to make less api calls
############################################
def make_querys():
    df = pd.read_csv("rag_sample_queries_candidates.csv")
    df.sort_values(["query_id", "baseline_rank"], inplace=True)
    temp = df.groupby("query_id")

    all_q = []
    for qid,group in temp:
        temp_q = []
        # print(qid)
        for idx, row in group.iterrows():
            # print(row)
            query =  str(row["query_id"]) + " " +  str(row["query_text"] + ": Canadate text: " + str(row["candidate_text"]))
            temp_q.append(query)
        # print(temp_q)
        # print(temp_q)
        all_q.append(temp_q)
    return all_q

def main():


    querys = make_querys()
    LLM_column = []
    for item in querys:

############################################
# making the query and the canadate text
############################################
        QUERY_TEXT = ""
        count = 0
        CANDIDATE_BLOCK = ""
        for t in item:
            o = t.split(":")
            Can = o[1] + o[-1]
            start = ABC[count] + " " + o[0]
            QUERY_TEXT += start + "\n"
            CANDIDATE_BLOCK += ABC[count] + Can + "\n"
            count += 1
        # print(QUERY_TEXT)
        # print(CANDIDATE_BLOCK)
#         b = f"""You are evaluating search results.

# For the query below, rate EACH candidate on a 0–5 scale (0 = not relevant, 5 = highly relevant).
# Return ONLY a minified JSON object mapping candidate_id -> score (float), e.g.: ("a": 4.0, "b": 0.5) but with braces and no ```json```

# <query>
# {QUERY_TEXT}
# </query>

# <candidates>
# {CANDIDATE_BLOCK}
# </candidates>

# """
#         print(b)

# i put the without ```json``` to have no errors
# i also only send out 20 requests so i dont hit my limit so i thought there was no need for testing

############################################
# asking the model
############################################
        answer = model.generate(messages=[{
            "role": "user", 
            "content": f"""You are evaluating search results.

For the query below, rate EACH candidate on a 0–100 scale (0 = not relevant, 100 = highly relevant).
Return ONLY a minified JSON object mapping candidate_id -> score (float), e.g.: ("a": 2, "b": 59) but with braces and no ```json```

<query>
{QUERY_TEXT}
</query>

<candidates>
{CANDIDATE_BLOCK}
</candidates>

"""
        }])
        print(answer.content)
        dictionary = json.loads(answer.content) 
        # print(dictionary)
        for key in dictionary:
            LLM_column.append(dictionary[key])

        print(f"Model returned answer: {answer.content}")
        time.sleep(5)
    # print(LLM_column)

############################################
#saving
############################################
    df = pd.read_csv("rag_sample_queries_candidates.csv")
    df.sort_values(["query_id", "baseline_rank"], inplace=True)
    df["LLM_Values"] = LLM_column
    df.to_csv("LLM_Output3.csv", index=False)


if '__main__' == __name__:
    main()