import requests

url = 'http://127.0.0.1:8080/upload'
files = {'myFile': open('/home/andres/Escritorio/CNN/hu/0a983cd2-0078-4698-a048-99ac01eb167a-1433917038889-1.7-f-04-hu.wav', 'rb')}

r = requests.post(url, files=files)
print(r.text)