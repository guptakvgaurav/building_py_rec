from flask import Flask, jsonify
from ml_algo.model import Model;
from ml_algo.model import AirFoil;
#from pyspark.sql.types import *;

#import pyspark;
#from pyspark import SparkConf, SparkContext, SQLContext

print('== [SERVER] -- Booting...');
linReg = Model()

app = Flask(__name__)

@app.route("/predict/sound")
def top10Recommendation():
	air_foil = AirFoil(800,0,0.3048,71.3,0.0026634);
	air_foil_in_target_format = air_foil.arange_in_tuple();
	prediction = linReg.predict(air_foil_in_target_format)
	print('== [Predictions] ==', prediction.head())
	return prediction.toJSON().first()


@app.route('/test')
def test():
	# create spark Session
	conf = SparkConf().setAppName("test").setMaster("local")
	sc = SparkContext(conf=conf)
	sqlContext = SQLContext(sc)

	# Approach#1
	tup = [(800,0,0.3048,71.3,0.0026634)]
	cols = ["Freq_Hz", "AoA_Deg", "Chord_m", "V_inf_mps", "displ_thick_m"];
	print(' ### Tuple is ', tup);

	#Approach#2
	schema = StructType([
		StructField("Freq_Hz", IntegerType(), False),
		StructField("AoA_Deg", IntegerType(), False),
		StructField("Chord_m", DoubleType(), False),
		StructField("V_inf_mps", DoubleType(), False),
		StructField("displ_thick_m", DoubleType(), False),
	]);
	print(' ### schema -->', schema.simpleString());
	# session = linReg.getSession(); # returns the spark session
	#print(' ### session -->', session.conf);

	# Approach 1
	df = sqlContext.createDataFrame(tup, schema)
	# Approach 2
	#df = session.createDataFrame(tup, schema)
	print(' ### data frame -->', df.toJSON())
	return df.toJSON()

if __name__ == "__main__":
	print('== [Server] Starting server...');
	app.run(debug=True, port=4000)

