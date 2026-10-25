import json, time, urllib.request
def post(ids):
    r = urllib.request.Request("http://localhost:8000/next", json.dumps({"ids": ids}).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(r))
for _ in range(240):
    try: urllib.request.urlopen("http://localhost:8000/docs"); break
    except Exception: time.sleep(5)
for ids in ([464, 3290, 318], [15496, 11, 616, 1438, 318, 257], [464, 3290, 318]):
    t = time.time(); out = post(ids); print(f"{ids} -> {out} ({time.time()-t:.1f}s)")
