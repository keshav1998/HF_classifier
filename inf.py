import requests

API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
headers = {"Authorization": "Bearer "}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Classify 'Cardiac Pathways Corporation' into either of these classes ['Organisation', 'Institution', 'Government Body', 'Other']:\n",
})
print(output[0]['generated_text'].split("\n")[-1])