'''
class cogito:
    def __init__(self):
        pass

    def train(self, env):
        pass

    def generate(self, env):
        output = self._gen(env)
        return output
'''


from labratory.type000 import cogito
from utils.twitterdata import get_random_tweet

if __name__ == '__main__':
    train_num = 10000
    c = cogito()
    for tweet in get_random_tweet(train_num):
        c.settrain('GAN')
        c.train(tweet)

    print(c.generate())