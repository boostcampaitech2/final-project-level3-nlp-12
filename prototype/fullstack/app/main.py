from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
import uvicorn
from predict import get_pipeline
from load_data import load_data, retrieve_comments
from tqdm import tqdm
from wordcloud import WordCloud
from pydantic import BaseModel
from io import BytesIO
from utils import *

app = FastAPI()
pipe = get_pipeline()
dataset = load_data()

app.router.redirect_slashes = False


class Data(BaseModel):
    comment: str


@app.get("/")
def hello_world():
    return {"hello": "world"}


@app.get("/inference/{keyword}")
def inference(keyword):
    results = []
    comments = retrieve_comments(keyword, dataset)

    for comment in tqdm(comments):
        output = pipe(comment)

        if output[0]["label"] == "none":
            continue

        else:
            results.append(
                {
                    "comment": comment,
                    "label": output[0]["label"],
                    "score": output[0]["score"],
                }
            )

    return results


@app.post("/wordcloud/", description="wordcloud를 생성합니다.")
def wordcloud(json: Data):
    json_compatible_item_data = jsonable_encoder(json)
    word_cloud = (
        WordCloud(
            font_path="MALGUN.TTF", background_color="white", width=500, height=500
        )
        .generate(json_compatible_item_data["comment"])
        .to_image()
    )

    img = BytesIO()
    word_cloud.save(img, "PNG")

    img.seek(0)
    return StreamingResponse(img, media_type="image/PNG")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
