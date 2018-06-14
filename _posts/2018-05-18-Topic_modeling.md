---
layout: post
title: Python code for topic modeling
subtitle: Analysis using research papers
---
<p> Topic modeling using LDA algorithm has many applications. In this post, I am sharing the code I used to find the research areas of an academic department. </p>

```python
'''
Section 1 scrapes mit eecs web page to create a list of faculty names in the department

'''
# ****** Section 1 *******
import requests
page = requests.get("http://www.eecs.mit.edu/people/faculty-advisors")
print page.status_code  # 200 code indicates successful download

from bs4 import BeautifulSoup
soup = BeautifulSoup(page.content.decode('ascii','ignore'), 'html.parser') #ignores any non-ascii character


# gather faculty names from the html class field-content card-title
names_tag=soup.find_all(class_="field-content card-title")
names =[]

#Change type<tag> to type<text>
for each in names_tag:
    names.append(each.get_text().encode('utf-8')) 
#print (names)
#print len(names)

# Format the names as LastName_FirstName for use in arxiv query

import re
faculty_list=[]
for each in names:
    first=each.split()[0]
    
    last = each.split()[-1]
    if last != first :
        name = last+'_'+first
    else:
        name =first
    faculty_list.append(name)
    

    
# ********* End of Section 1 ************


from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

all_abstract =[]
for name in faculty_list:
    url='http://export.arxiv.org/api/query?search_query=au:'+name
    arxiv_page = requests.get(url)
    soup = BeautifulSoup(arxiv_page.content, 'html.parser')
    abstract_tag=soup.find_all('summary') 
    
    
    # Data cleaning
    for each in abstract_tag:
        abstract_list_individual =[]
        abstract_list_individual.append(each.get_text().encode('utf-8'))
        abstract_individual = str(abstract_list_individual)
        abstract_clean = re.sub(r"\\n|\\|(' )|(',  )|'",' ',abstract_individual)
  
        # Tokenizer
        tokenizer= RegexpTokenizer('\w+')
        tokens_all = tokenizer.tokenize(abstract_clean)
    
        # Remove stop words
        stop_words = set(stopwords.words("english"))
        tokens_without_stops =[]

        for w in tokens_all:
            if w not in stop_words:
                tokens_without_stops.append(w)
        tokens_without_stops
    
        # Stemming
        p_stemmer = PorterStemmer()
        tokens_stemmed=[]
        for i in tokens_without_stops:
            tokens_stemmed.append(p_stemmer.stem(i).encode('utf8')) 
        all_abstract.append(tokens_stemmed)


import gensim
from gensim import corpora, models
dictionary = corpora.Dictionary(all_abstract)

corpus = [dictionary.doc2bow(text) for text in all_abstract]
print len(corpus)
print(corpus[0])

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary, passes=3)
print(ldamodel.print_topics(num_topics=2, num_words=3))

'''
from sklearn.decomposition import NMF, LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components = 40, max_iter = 10, random_state=1)
lda.fit(corpus)
'''
```

       



