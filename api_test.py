import requests

url = 'http://0.0.0.0:5000/clusterize'

payload = {
    'x': 2.8,
    'y': 8.5,
}

if __name__ == '__main__':
    r = requests.post(url=url, json=payload)
    print(r.text)