import requests

response = requests.get("https://randomuser.me/api")
print(response.status_code) 
data=response.json()

gender=data['results'][0]['gender']
print(gender)
title=data['results'][0]['name']['title']
first_name=data['results'][0]['name']['first']
last_name=data['results'][0]['name']['last']
print(f'Title is {title}. First name: {first_name} Last Name: {last_name}')
street_number=data['results'][0]['location']['street']['number']

street_name=data['results'][0]['location']['street']['name']
#                          Key       Value
city=data['results'][0]['location']['city']

state=data['results'][0]['location']['state']

country=data['results'][0]['location']['country']

postcode=data['results'][0]['location']['postcode']

print(f"Street Number: {street_number} Street Name: {street_name} City: {city} State: {state} Country: {country} Postcode: {postcode}")
#key and value
email=data['results'][0]['login']['username']
print(email)