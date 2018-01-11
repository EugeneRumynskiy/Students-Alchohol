# Students-alchohol model. With prediction API on Flask. Deployed to heroku as app

## Roadmap
	
|    |      Goal      |  Tool |
| ---------- | ------------- | ------ |
| 1 |  Загрузить данные локально	 			| pandas, numpy	|
| 2 |  Обучить модель                       			| sklearn          	|
| 3 |  Сохранить модель  					| dill, pickle 		|
| 4 |  Создать API на Flask 		 			| flask     		|
| 5 |  Поднять API локально на сервере gunicorn | gunicorn 		|
| 6 |  Поднять API в виде приложения на heroku			| HerokuCLI		|


## Reference
Гайд основан на двух подробных статьях.
### [Tutorial to deploy Machine Learning models in Production as APIs (using Flask)](https://www.analyticsvidhya.com/blog/2017/09/machine-learning-models-as-apis-using-flask/)
	
	Roadmap:
	1. Options to implement Machine Learning models
	2. What are APIs?
	3. Python Environment Setup & Flask Basics
	4. Creating a Machine Learning Model
	5. Saving the Machine Learning Model: Serialization & Deserialization
	6. Creating an API using Flask

Туториал мне понравился подробностью и качеством кода. По итогам мы сможем запустить API локально. Хочется посмотреть как будет с облаком, поэтому я нашёл ещё одну статью на эту тему.


###[Create a scikit-learn based prediction webapp using Flask and Heroku](https://xcitech.github.io/tutorials/heroku_tutorial/)
	
	Roadmap:
	1. Create the Model in Python using Scikit-learn
	2. Saving the Model using Pickle
	3. Creating the Flask webapp
	4. Deploy the app to Heroku

Качетсво кода здесь хуже, но по итогам наш API  будет висеть в облаке. Можно будет сделать что-то в роде - 

	link = """https://students-and-alcohol.herokuapp.com/predict"""
	resp = requests.post(link, data = json.dumps(data), headers= header)
	
Вместо беспонтового

	link = """http://0.0.0.0:8000/predict"""
	resp = requests.post(link, data = json.dumps(data), headers= header)
