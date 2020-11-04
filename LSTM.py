import pandas
from sklearn import preprocessing


# 為了避免原始數據太大或是太小沒有統一的範圍而導致 LSTM 在訓練時難以收斂，
# 我們以一個最小最大零一正規化方法對數據做正規化
def normalize(df):
	
	newdf = df.copy()
	min_max_scaler = preprocessing.MinMaxScaler()

	norm_columns = ['open','high','low','close']
	
	for col in norm_columns:
		newdf[col] = min_max_scaler.fit_transform(newdf[col].values.reshape(-1,1))

	return newdf


# 資料分割為training validation and test 

valid_size_pct = 0.1
test_size_pct = 0.1

def dataloader(df, time_frame, valid_size_pct=0.1, test_size_pct= 0.1):
	df_matrix = df.as_matrix()
	data = []

	# create all possible sequences of length seq_len
	for idx in range(len(df_matrix)-time_frame):
		data.append(df_matrix[idx: idx + time_frame])