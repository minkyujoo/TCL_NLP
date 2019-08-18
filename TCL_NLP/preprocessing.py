import pandas as pd

#train_data from csv
train_data = [['어쩌구 저쩌구를 요청 드립니다.', 1], #Request
          ['어쩌구 저쩌구 \r\n답변 부탁 드립니다.', 2], #Reply
          ['그래서 \r\n모임 요청 드립니다.',3], #Meeting
          ['감사합니다.', 4]] #Other

columns = ['body', 'intent']

df = pd.DataFrame(train_data, columns=columns)

# split training data
X, y = df['body'], df['intent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state =1234)

