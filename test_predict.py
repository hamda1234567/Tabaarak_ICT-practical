import json
import urllib.request

payload = {"studytime":2,"failures":1,"schoolsup":"yes","famsup":"yes","activities":"no","higher":"yes","internet":"yes","famrel":4,"health":3,"absences":8}
req = urllib.request.Request('http://127.0.0.1:3200/predict', data=json.dumps(payload).encode('utf-8'), headers={'Content-Type':'application/json'}, method='POST')
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        print('STATUS', resp.status)
        print(resp.read().decode())
except Exception as e:
    print('ERROR', repr(e))
