from django.test import Client
client = Client()
response = client.get("/main-page/")
print(response.status_code)
