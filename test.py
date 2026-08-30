
import requests
prompt = "write a short story about a robot learning to"
response= requests.post(
  "http://localhost:11434/api/generate",
    json={ 
"model":"qwen2.5:3b",
"stream":False,
  "prompt":prompt
    }

)
answer= response.json()["response"]
print(answer)