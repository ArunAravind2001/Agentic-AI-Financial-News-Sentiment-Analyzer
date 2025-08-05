

## news_agent

import pandas as pd

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import spacy

from newsapi import NewsApiClient


API_KEY = "YOUR_API_KEY"

MODEL_NAME = "ProsusAI/finbert"

NEWS_PHRASE = "expansion"

NUM_ARTICLES = 50  

LANGUAGE = "en"


nlp = spacy.load("en_core_web_sm")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.eval()

news_api = NewsApiClient(api_key=API_KEY)

def fetch_business_news(query, n_news=NUM_ARTICLES):

    response = news_api.get_everything(

        q=query,

        language=LANGUAGE,

        sort_by='publishedAt',

        page_size=min(n_news, 100),

        page=1

    )

    articles = response.get("articles", [])

    cleaned = [{

        "title": art.get("title"),

        "description": art.get("description"),

        "publishedAt": art.get("publishedAt"),

        "source": art.get("source", {}).get("name")

    } for art in articles]

    return pd.DataFrame(cleaned)

def analyze_sentiment(df):

    if df.empty or 'title' not in df.columns:

        print("Warning: No titles found in the dataframe")

        return df

    titles = (df["title"].fillna("") + ". " + df["description"].fillna("")).tolist()

    if not titles:

        print("Warning: No valid titles found")

        return df

    inputs = tokenizer(titles, padding=True, truncation=True, max_length=512, return_tensors='pt')

    with torch.no_grad():

        outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    sentiment_df = pd.DataFrame({

        'title': df["title"].tolist(),

        'positive': probs[:, 0].tolist(),

        'negative': probs[:, 1].tolist(),

        'neutral': probs[:, 2].tolist()

    })

    sentiment_df["sentiment_label"] = sentiment_df[["positive", "neutral", "negative"]].idxmax(axis=1)

    sentiment_df["sentiment_score"] = sentiment_df.apply(lambda row: row[row["sentiment_label"]], axis=1)

    return df.merge(sentiment_df, on="title", how="left")

def extract_companies(text):

    if pd.isna(text):

        return []

    doc = nlp(text)

    return [ent.text for ent in doc.ents if ent.label_ == "ORG"]



def run_news_sentinel_agent():

    print("🔍 Fetching business news...")

    news_df = fetch_business_news(NEWS_PHRASE, NUM_ARTICLES)

    if news_df.empty:

        print("❌ No news articles found.")

        return news_df

    print(f"📰 Found {len(news_df)} articles.")

    print("🧠 Analyzing sentiment...")

    news_df = analyze_sentiment(news_df)

    if 'sentiment_label' not in news_df.columns:

        print("❌ Sentiment analysis failed.")

        return news_df

    print("✅ Filtering for positive sentiment...")

    news_df = news_df[

        (news_df["sentiment_label"] == "positive") &

        (news_df["sentiment_score"] > 0.3)

    ]

    print("🏷️ Extracting company names from headlines...")

    news_df["companies"] = news_df["title"].apply(extract_companies)

    print("✅ Done. Here's the result:")

    print(news_df[["title", "sentiment_label", "sentiment_score", "companies"]])

    return news_df



if __name__ == "__main__":
    final_result = run_news_sentinel_agent()

    final_result = final_result.sort_values(by="sentiment_score", ascending=False)

    all_companies = []

    for i in final_result["companies"]:
        for j in i:
            all_companies.append(j)

    top_10 = list(set(all_companies))[:10]




# In[2]:


from typing import TypedDict
import pandas as pd

class AgentState(TypedDict):
    df:pd.DataFrame
    query:str
    top_companies:list


# In[5]:


def news_scraper(state: AgentState) -> AgentState:
    query=state.get("query","expansion")
    df=fetch_business_news(query)
    return{"df":df, "query":query}


# In[6]:


def sentiment_analyzer(state: AgentState) ->AgentState:
    df=analyze_sentiment(state["df"])
    return {"df":df, "query":state["query"]}


# In[7]:


def company_extractor(state: AgentState) -> AgentState:
    df = state["df"]
    df["companies"] = df["title"].apply(extract_companies)
    return {"df": df, "query": state["query"]}


# In[8]:


def aggregator(state: AgentState) -> AgentState:
    df = state["df"]
    df = df[df["sentiment_label"] == "positive"]
    all_companies = [company for row in df["companies"] for company in row]
    top_10 = list(set(all_companies))[:10]
    return {"df": df, "query": state["query"], "top_companies": top_10}


# In[9]:


def responder(state: AgentState) -> AgentState:
    print("✅ Top Bullish Companies:")
    for company in state["top_companies"]:
        print("➡️", company)
    return state  # final output


# In[10]:


from langgraph.graph import StateGraph, END

builder = StateGraph(AgentState)

# Add nodes
builder.add_node("news_scraper", news_scraper)
builder.add_node("sentiment_analyzer", sentiment_analyzer)
builder.add_node("company_extractor", company_extractor)
builder.add_node("aggregator", aggregator)
builder.add_node("responder", responder)

# Set entry and flow
builder.set_entry_point("news_scraper")
builder.add_edge("news_scraper", "sentiment_analyzer")
builder.add_edge("sentiment_analyzer", "company_extractor")
builder.add_edge("company_extractor", "aggregator")
builder.add_edge("aggregator", "responder")
builder.add_edge("responder", END)

# Compile
graph = builder.compile()


# In[23]:





# In[24]:





# In[ ]:





