from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
import uvicorn

from predict import load_model, inference, load_dataloader
from database import run_db, insert2db
from load_data import load_data, retrieve_comments

from wordcloud import WordCloud, STOPWORDS
from io import BytesIO
from pydantic import BaseModel


app = FastAPI()
dataset = load_data()
model, classifier = load_model()
# evidence = run_db()

class Data(BaseModel):
    comment: str
    keyword: str

@app.get('/')
def hello_world():
    return {'hello': 'world'}


@app.get('/get_sample/{keyword}')
def get_sample(keyword):
    data = retrieve_comments(keyword, dataset)
    inf_dataloader = load_dataloader(data)
    results = inference(model, classifier, inf_dataloader)
    
    docs = []
    for i, res in enumerate(results):
        if res == 0:
            continue
        else:
            docs.append({
                "keyword": keyword,
                'user_id': data[i]['user_id'],
                'comment': data[i]['comment'],
                "label": 'hate',
                'site_name': data[i]['site_name'],
                'site_url': data[i]['site_url'],
                'commented_at': data[i]['commented_at']
            })
  
    # if docs:
    #     insert2db(keyword, docs, evidence)
  
    return docs[:5]

# @app.get('/get_count/{keyword}')
# def get_count(keyword):
#     cnt = len(evidence.find({'keyword': keyword}))
#     return {'count': cnt}

@app.post("/wordcloud/", description="wordcloud를 생성합니다.")
def wordcloud(json: Data):
    json_compatible_item_data = jsonable_encoder(json)

    stopwords = set(STOPWORDS)
    stopwords.add(json_compatible_item_data['keyword'])
    stopwords.add(json_compatible_item_data['keyword']+'는')
    stopwords.add(json_compatible_item_data['keyword']+'가')
    stopwords.add(json_compatible_item_data['keyword']+'도')
    word_cloud = (
        WordCloud(
            width=500, height=500, stopwords=stopwords
        )
        .generate(json_compatible_item_data["comment"])
        .to_image()
    )

    img = BytesIO()
    word_cloud.save(img, "PNG")

    img.seek(0)
    return StreamingResponse(img, media_type="image/PNG")


if __name__ == '__main__':
    uvicorn.run('main:app', host="0.0.0.0", port=8000, reload=True)  
    # result = get_sample('유재석')
    # print(result)
