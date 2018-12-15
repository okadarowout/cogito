import json, config
from requests_oauthlib import OAuth1Session
import datetime, time

CK = config.CONSUMER_KEY
CS = config.CONSUMER_SECRET
AT = config.ACCESS_TOKEN
ATS = config.ACCESS_TOKEN_SECRET
twitter = OAuth1Session(CK, CS, AT, ATS) #authorization

url = "https://api.twitter.com/1.1/statuses/update.json" # tweet post endpoint

tweet = 'Hello World!'

params = {"status" : tweet}

res = twitter.post(url, params = params) #post送信

if res.status_code == 200: #正常投稿出来た場合
    print("Success.")
else: #正常投稿出来なかった場合
    print("Failed. : %d"% res.status_code)

url = "https://api.twitter.com/1.1/statuses/user_timeline.json" #タイムライン取得エンドポイント

res = twitter.get(url)

limit = res.headers['x-rate-limit-remaining'] #リクエスト可能残数の取得
reset = res.headers['x-rate-limit-reset'] #リクエスト叶残数リセットまでの時間(UTC)
sec = int(res.headers['X-Rate-Limit-Reset']) - time.mktime(datetime.datetime.now().timetuple()) #UTCを秒数に変換

print ("limit: " + limit)
print ("reset: " +  reset)
print ('reset sec:  %s' % sec)

def make_params():
    query = 'AI min_replies:5 min_retweets:0 min_faves:1 exclude:retweets lang:en'
    params = {'q': query,
              'count': 20}
    return params

def make_params():
    query = 'running min_replies:0 min_retweets:1 min_faves:1 exclude:retweets lang:en'
    params = {'q': query,
              'count': 100}
    return params


def search_tweet(api, params):
    url = 'https://api.twitter.com/1.1/search/tweets.json'
    req = api.get(url, params=params)

    result = []
    if req.status_code == 200:
        tweets = json.loads(req.text)
        result = tweets['statuses']        
    else:
        print("ERROR!: %d" % req.status_code)
        result = None

    assert(len(result) > 0)

    return result

res = search_tweet(twitter, make_params())

print(len(res))

for dic in res:
    print(dic['text'])
