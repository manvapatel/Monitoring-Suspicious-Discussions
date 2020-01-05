TITLE: Suspicious Discussion Detection

- Description: As internet technology had been increasing more and more, it has led to many legal and illegal activities. This provides an effective channel for illegal activities such as threatening and abusive messages. We propose a system which will analyse online plain text sources from few selected discussion forums like twitter and will classify the text into different groups and decide whether the post is legal or illegal.

- Dataset Information: We have done aspect based analysis and for that, we prepared two types of training datasets. The dataset used for aspect generation contains of 16000 tweets. This dataset consists of two columns: Tweets and Aspect. Further, the we have used another dataset for classifying the text. This dataset consists of a csv of 2500 tweets having the following properties: Tweet_id, Tweet, Aspect and Suspiciousness (0 for suspicious tweet and 1 for tweet that is not suspicious).

- Requirements: There are some general library requirements for the project as follows:

  • Keras

  • Tensorflow

  • Scikit-learn

  • Numpy

  • Pandas

  • Scrapy

Further, there are program related dependencies as follows: • Crawler: MongoDB is required to store the scraped data • Aspect Generation: Uses Magnitude for vector embeddings. File can be downloaded from the following link: https://github.com/plasticityai/magnitude • Suspicious Detection: Download pre-trained Twitter word vectors, extract the glove.twitter.27B.zip and glove.42B.300d.zip to the root directory.

- Usage:

Scraping: Using scrapy, we have crawled the data of Twitter using Hashtags. The data is stored in MongoDB and it contains attributes like Username, TweetID, Tweet, URL, Hashtags,etc. The command to run the program is given in twitter_run.bat file in TweetScraper. Further, we have also crawled the data of blogspot and wordpress. It contains the following attributes: Date, Author, Title, Paragraph, Comment-name and Comment-content.

- Preprocessing: The data stored in the database is pre-processed using twitter_preprocess.py. It removes blank space, #, @ and urls from the data and stores this data in a csv file. The code is run as follows: python preprocess.py

- Aspect Generation: A CNN model is used to extract the aspects or categories of the tweets and sentences. The output of the above code is tweet and its aspect. The code is run as follows: python aspect_cnn.py. The validation accuracy of this model is 89.5.

- Suspicious detection: A LSTM model is used to classify the text into the following two categories: 1 for not suspicious and 0 for suspicious. The input given to this code consists of tweet and its aspect which is generated using the above code. The output is either 0 or 1. This code is run as follows: python ram_new.py. The validation accuracy of this model is 72.

- Limitations: There are various forms of data on social media. This model only works on textual data and not on other forms of data such as images, videos, gifs, etc.
