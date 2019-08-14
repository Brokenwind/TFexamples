import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def transition_probability(vec_arr):
    n = vec_arr.shape[0]
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = cosine_similarity(vec_arr[i].reshape(1, -1),
                                        vec_arr[j].reshape(1, -1))[0][0]
            m[j, i] = m[i, j]
    for i in range(n):
        if not np.count_nonzero(m[i]):
            m[i] = np.full((1, n), 1 / n)
    return preprocessing.normalize(m, norm='l1')


def score(m, mu=0.85, epsilon=0.0001, n=50):
    score = np.full((m.shape[0], 1), 1)
    for _ in range(n):
        temp = score.copy()
        score = mu * np.mat(m).T * score + (1 - mu) / m.shape[0]
        if max(abs(temp - score)) < epsilon:
            break
    return score


def summary(sentences, n=5):
    # take the top n sentences in scores
    vectorizer = CountVectorizer(binary=True)
    # vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(sentences)
    m = transition_probability(x.toarray())
    scores = score(m).A
    sents = []
    for i in np.argsort(-scores, axis=0).flatten()[:n]:
        sents.append(sentences[i])
    return sents


text = '''华商报铜川讯 “约150米的距离，被密密麻麻设置了41条减速带。
昨日，王先生驾车途经铜川市新区一小区附近辅道时，倒吸了一口凉气。远远望去，地上的一条条减速带，仿佛一大型“搓衣板”。
斜坡路段铺设41条减速带
昨日，华商报记者实地走访发现，在铜川市新区龙记翡翠城小区东门一侧的辅道是一段斜坡，路口至小区东门约150米的距离布满了大小不一的减速带，细数发现共有41条。
其中部分减速带间隔仅1米，剩余减速带间隔从2 - 10
米不等。而这还未算上该小区东门正前方的减速带。有附近商户表示，铺设的减速带有两年了，他曾骑电动车从这条路通过，一路受尽颠簸。
“这里是史上最强搓衣板路。”另一位商户表示，平日一些业主将车停放在道路一侧，而近期，“物业利用石墩将道路封锁，基本上就没有车辆通行了。”
业主有人支持
有人觉得小题大做
辅道为何铺设如此多的减速带?该小区业主均说不知情。
该小区一位刘姓业主说，道路的坡度谈不上陡峭，数条减速带就可以了，铺设这么多的减速带未免过于小题大做。加上小区安全通道就位于这条路的旁边，这样遇到紧急情况会影响到消防通行。
“减速带有不可替代的功效。”另一位业主则认为，这条斜坡道路若没有减速带，就是通往小区的重要通道。
而道路下方正对的就是广场，夏日人流量车流量肯定非常大，这些减速带正好可以减少车辆通行、减缓车辆行驶速度，从而提高业主与小区幼儿的出行安全。 '''
sentences = text.split(sep='\n')
print(sentences[0])
print(summary(sentences))
