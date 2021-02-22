---
layout: post
title:  "YouTube API"
categories: notebook
author:
- Jenei Bendegúz
excerpt: Playlist duration calculation with the YouTube API
---

<a href="https://colab.research.google.com/github/jben-hun/colab_notebooks/blob/master/youtubeAPI.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Calculate the total duration of a youtube playlist by getting all video ids it contains, and accumulating their durations**

Refer to the link below to learn how to enable the API. Note that at the time of writing this notebook, the code examples in the official guide use the OAuth 2.0 method to authenticate, while this notebook simply uses an API key (option "a" in the guide). OAuth 2.0 is only required when accessing account related content.



*   Official API intro guide: <https://developers.google.com/youtube/v3/quickstart/python>


```python
import getpass
import googleapiclient.discovery
import numpy as np
import pandas as pd
from datetime import timedelta
```


```python
api_key = getpass.getpass("Enter your API key: ")
youtube = googleapiclient.discovery.build("youtube", "v3",
                                          developerKey=api_key)
```

    Enter you API key: ··········
    

**Performing the neccessary requests**



Carefully specifying the "part" parameter of the requests is important to conserve quoatas, as it specifies which info we want in our response. Set it as narrow/concise as possible.



*   <https://developers.google.com/youtube/v3/getting-started#partial>


```python
playlist_id = input("Enter a youtube playlist id: ")
```

    Enter a youtube playlist id: PLGVZCDnMOq0rLLb519Ah3EntCUAAHPnfU
    


```python
# limit used for pagination
page_size = 50

# get video identifiers from playlist identifier (page_size at a time)
videoIds = []
query = dict(
    part="contentDetails", maxResults=page_size, playlistId=playlist_id)
while True:
  request = youtube.playlistItems().list(**query)
  result = request.execute()
  videoIds += [i["contentDetails"]["videoId"] for i in result["items"]]
  if "nextPageToken" in result:
    query["pageToken"] = result["nextPageToken"]
  else:
    break

# get video metadata by video identifiers (page_size at a time)
durations = []
while videoIds:
  idList = videoIds[:page_size]
  videoIds = videoIds[page_size:]
  request = youtube.videos().list(
    part="contentDetails", id=",".join(idList))
  result = request.execute()
  durations += [r["contentDetails"]["duration"] for r in result["items"]]
```

**Playlist duration calculation**


```python
timedeltas = pd.to_timedelta(pd.Series(durations).str.slice(start=2))

print(f"playlist: https://www.youtube.com/playlist?list={playlist_id}\n"
      f"videos: {len(timedeltas)}\n"
      f"mean_playtime: {timedeltas.mean()}\n"
      f"total_playtime: {timedeltas.sum()}")
```

    playlist: https://www.youtube.com/playlist?list=PLGVZCDnMOq0rLLb519Ah3EntCUAAHPnfU
    videos: 52
    mean_playtime: 0 days 00:46:44.961538461
    total_playtime: 1 days 16:30:58
    
