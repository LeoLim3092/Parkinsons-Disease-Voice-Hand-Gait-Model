import requests
import os
import urllib.parse

FIXED_TRANSCRIPT = (
    "有一回 北風和太陽正在爭論誰的能耐大 爭來爭去 就是分不出個高低來 "
    "這會兒 來了個路人 他身上穿了件厚大衣 他們倆就說好了 "
    "誰能先叫這個路人把他的厚大衣脫下來 就算誰比較有本事 "
    "於是 北風就拚命地吹 怎料 他吹得越厲害 那個路人就把大衣包得越緊 "
    "最後 北風沒辦法 只好放棄 過了一陣子 太陽出來了 "
    "他火辣辣地曬了一下 那個路人就立刻把身上的厚大衣脫下來 "
    "於是 北風只好認輸了 他們倆之間還是太陽的能耐大"
)

def score_pronunciation(wav_path, server_url="http://localhost:8899"):
    # Decode URL-encoded path
    wav_path = urllib.parse.unquote(wav_path)

    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    # POST audio and transcript to scoring server
    with open(wav_path, 'rb') as audio_file:
        response = requests.post(
            f"{server_url}/scoring",
            data={
                'text': FIXED_TRANSCRIPT,
                'filters': 'utterance,cm_word_text,cm_word_timberScore'
            },
            files={
                'data': audio_file
            },
        )

    if response.status_code != 200:
        raise Exception(f"Server error: {response.status_code} - {response.text}")

    result = response.json()
    uid = result.get('uid')
    if not uid:
        raise Exception("No UID returned from scoring server.")

    # Fetch the score JSON
    json_url = f"{server_url}/score_json?json_name={uid}.json"
    json_response = requests.get(json_url)

    if json_response.status_code != 200:
        raise Exception(f"Could not fetch score JSON: {json_response.status_code}")
        
    results = json_response.json()

    return results["cm"]["score"]
