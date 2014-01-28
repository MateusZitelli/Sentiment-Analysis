# -*- coding: cp1252 -*-
import twitter # api do twitter


class Tweets:

    def __init__(self,subject,qtd):
        self.api = twitter.Api()
        self.subject = subject
        self.qtd = qtd

    def getTweets(self,query):
        pages = self.qtd/100
        tweets = []
        for p in range(1,pages+1):
            statuses = self.api.GetSearch(query.decode('latin-1'),per_page=100,lang='pt',page=p)
            tweets += map(lambda x: x.text.encode('utf8'), statuses)
        return tweets

    def get(self):
        # buscar por tweets contendo o termo do assunto requisitado
        query = '%s' % ( self.subject )        
        tweets = self.getTweets(query)

        return tweets
