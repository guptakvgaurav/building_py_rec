import findspark;
findspark.init()

import pyspark;
from pyspark.sql import SparkSession;
from pyspark.ml.regression import LinearRegression;
from pyspark.ml.feature import VectorAssembler;
from pyspark.sql.types import *;
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.feature import StandardScaler

X_Cols = ["Freq_Hz", "AoA_Deg", "Chord_m", "V_inf_mps", "displ_thick_m"]


class Model:
	
	spark = None;
	model = None;
	airfoil_assembler = None;
	entire_Set = None;

	std_scaler = None;

	def getSession(self):
		return self.spark

	def __init__(self):
		print('== [Model] Creating spark session...');
		self.spark = SparkSession.builder.appName('lin_reg_api').getOrCreate();
		#self.spark = SparkSession.newSession()
		print('== [Model] spark version', self.spark.version)
		print('== [Model] Loading model...');
		self.model = LinearRegressionModel.load('model_lin_reg')
		print('== [Model] Loading complete...');

		self.entire_Set = self.spark.read.csv('./airfoil_self_noise.csv', header=True, inferSchema=True);

		# define transformers...
		self.airfoil_assembler = VectorAssembler(inputCols=X_Cols, outputCol='_features')
		freq_scaler = StandardScaler(inputCol="_features", outputCol="features");

		tuned_input_vec = self.airfoil_assembler.transform(self.entire_Set).select('_features');
		self.std_scaler = freq_scaler.fit(tuned_input_vec)
		return ;

	def _getSchema(self):
		schema = StructType({
			StructField("Freq_Hz", IntegerType(), False),
			StructField("AoA_Deg", IntegerType(), False),
			StructField("Chord_m", DoubleType(), False),
			StructField("V_inf_mps", DoubleType(), False),
			StructField("displ_thick_m", DoubleType(), False),
		});
		return schema

	def _prepare_df(self):
		schema = self._getSchema();
		return df;

	def assemble(self, tup):
		schema = self._getSchema();
		df = self.spark.createDataFrame(tup, schema)
		#print('== [Model] Created df looks like', df.head())
		assembled_vector = self.airfoil_assembler.transform(df);
		return assembled_vector.select("features");

	def _standardize_input(self, inputVector):

	    # transform data
		tuned_input_vec = self.airfoil_assembler.transform(inputVector).select('_features');
		tuned_input_vec = self.std_scaler.transform(tuned_input_vec);

		return tuned_input_vec.select('features');


	def getSample(self):
		# generate sample...
		sample = self.entire_Set.sample(False, 0.1).limit(1);
		print(sample.show())
		return sample, self._standardize_input(sample);

	def predict(self, airfoil):
		#assembled_vector = self.assemble(tup=airfoil)
		#print('== [Model] Assembled vec looks like -', assembled_vector.head())
		sample, normalized_sample = self.getSample();
		print('=== Sample looks like - ', sample.show());
		print('=== Normalized sample looks like - ', normalized_sample.show());
		return self.model.transform(normalized_sample)


class AirFoil:

	freq = None;
	AoA = None;
	Chord_M = None;
	V_inf = None;
	disp_thick = None;

	def __init__(self, f, a, c, v, d):
		self.freq = f;
		self.AoA = a;
		self.Chord_M = c;
		self.V_inf = v;
		self.disp_thick = d;
		return

	def arange_in_tuple(self):
		tup = [(self.freq, self.AoA, self.Chord_M, self.V_inf, self.disp_thick)];
		print('== [Model] Airfoil is', tup)
		return tup;