import requests

response = requests.get("https://www.flipkart.com/just97-8-hole-bubble-maker-gatling-gun-machine-toy-kids-solution-a50-water/p/itm482df2f1cd0da?pid=TWPH2VWWRZVGKU9Z&lid=LSTTWPH2VWWRZVGKU9ZUWESTJ&marketplace=FLIPKART&store=tng%2Fsv3&srno=b_1_13&otracker=browse&fm=organic&iid=8c5f0556-a40c-4b39-bb02-99c0ee214714.TWPH2VWWRZVGKU9Z.SEARCH&ppt=browse&ppn=browse&ssid=9s04gtckmo0000001735296663865")
print(response.status_code) 
data=response.text
desc=data['meta'][0]['content']
print(desc)