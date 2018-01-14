# testing server
import json
import requests
import pandas as pd
import numpy as np

"""Setting the headers to send and accept json responses
"""
header = {'Content-Type': 'application/json', 'Accept': 'application/json'}

"""Making predictions for por students. It's not right, but just for example.
"""
test_df = pd.read_csv('../data/student-por.csv', sep=";")
#numeric_cols = ["age", 'Medu', "Fedu", "traveltime", "studytime", "failures",
#                "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2"]

data = test_df.to_json(orient='records')

"""POST <url>/predict
"""
link = """https://students-and-alcohol.herokuapp.com/predict"""
#link = """http://0.0.0.0:8024/predict"""
resp = requests.post(link, data=json.dumps(data), headers=header)
print(resp.status_code)

"""The final response we get is as follows:
"""
predictions = resp.json()
predictions = pd.read_json(predictions["predictions"])
predictions.columns = ["predicted"]

print(np.abs(test_df["G3"] - predictions["predicted"]).mean())
#print(predictions["predicted"])

