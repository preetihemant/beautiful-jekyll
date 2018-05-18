---
layout: post
title: Python code for topic modeling
subtitle: Analysis using research papers
---
<p> Topic modeling using LDA algorithm has many applications. In this post, I am sharing the code for how I used it to find the research areas of an acedemic department. </p>

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
    names.append(each.get_text().encode('utf-8'))  #encoding gets rid of the u' in front of every word
#print (names)
#print len(names)

# Format the names as LastName_FirstName for use in arxiv query
# To DO - handle names formatted unusually
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
    
#print (faculty_list)

    
# ********* End of Section 1 ************

#faculty_list = ['Abelson_Hal', 'Adalsteinsson_Elfar', 'Agarwal_Anant', 'Akinwande_Akintunde', 'Alizadeh_Mohammad', 'Amarasinghe_Saman', 'Antoniadis_Dimitri', 'Arvind', 'Baggeroer_Arthur', 'Balakrishnan_Hari', 'Baldo_Marc', 'Barzilay_Regina', 'Belay_Adam', 'Berger_Bonnie', 'Berggren_Karl', 'Berners-Lee_Tim', 'Bertsekas_Dimitri', 'Berwick_Robert', 'Bhatia_Sangeeta', 'Boning_Duane', 'Braida_Louis', 'Bresler_Guy', 'Broderick_Tamara', 'Brooks_Rodney', 'Bulovic_Vladimir', 'Carbin_Michael', 'Chan_Vincent', 'Chandrakasan_Anantha', 'Chlipala_Adam', 'Chuang_Isaac', 'Clark_David', 'Corbat_Fernando', 'Dahleh_Munther', 'Daniel_Luca', 'Daskalakis_Constantinos', 'Davis_Randall', 'Alamo_Jess', 'Demaine_Erik', 'Dennis_Jack', 'Devadas_Srini', 'DeWitt_David', 'Durand_Fredo', 'Emer_Joel', 'Eng_Tony', 'Englund_Dirk', 'Fonstad_Clifton', 'Forney_David', 'Freeman_Dennis', 'Freeman_William', 'Fujimoto_James', 'Gallager_Robert', 'Gifford_David', 'Goldwasser_Shafi', 'Golland_Polina', 'Gray_Martha', 'Grodzinsky_Alan', 'Guttag_John', 'Hagelstein_Peter', 'Han_Jongyoon', 'Han_Ruonan', 'Heldt_Thomas', 'Horn_Berthold', 'Hoyt_Judy', 'Hu_Qing', 'Indyk_Piotr', 'Ippen_Erich', 'Jaakkola_Tommi', 'Jackson_Daniel', 'Jaillet_Patrick', 'Jegelka_Stefani', 'Kaelbling_Leslie', 'Karger_David', 'Kassakian_John', 'Katabi_Dina', 'Kellis_Manolis', 'Kolodziejski_Leslie', 'Kong_Jing', 'Kraska_Tim', 'Lampson_Butler', 'Lang_Jeffrey', 'Lee_Hae-Seung', 'Leeb_Steven', 'Leiserson_Charles', 'Lim_Jae', 'Liskov_Barbara', 'Liu_Luqiao', 'Lo_Andrew', 'Lozano-Prez_Toms', 'Lu_Timothy', 'Lynch_Nancy', 'Madden_Samuel', 'Madry_Aleksander', 'Magnanti_Thomas', 'Mark_Roger', 'Matusik_Wojciech', 'Mdard_Muriel', 'Megretski_Alexandre', 'Meyer_Albert', 'Micali_Silvio', 'Miller_Rob', 'Mitter_Sanjoy', 'Morris_Robert', 'Moses_Joel', 'Mueller_Stefanie', 'Oppenheim_Alan', 'Orlando_Terry', 'Ozdaglar_Asuman', 'Palacios_Toms', 'Parrilo_Pablo', 'Peh_Li-Shiuan', 'Perreault_David', 'Polyanskiy_Yury', 'Ram_Rajeev', 'Rinard_Martin', 'Rivest_Ronald', 'Rubinfeld_Ronitt', 'Rupp_Jennifer', 'Rus_Daniela', 'Sanchez_Daniel', 'Schindall_Joel', 'Schmidt_Martin', 'Shah_Devavrat', 'Shapiro_Jeffrey', 'Shavit_Nir', 'Shulaker_Max', 'Shun_Julian', 'Smith_Henry', 'Sodini_Charles', 'Solar-Lezama_Armando', 'Solomon_Justin', 'Sontag_David', 'Sra_Suvrit', 'Stonebraker_Michael', 'Stultz_Collin', 'Sussman_Gerald', 'Sze_Vivienne', 'Szolovits_Peter', 'Tedrake_Russell', 'Terman_Christopher', 'Tidor_Bruce', 'Torralba_Antonio', 'Tsitsiklis_John', 'Uhler_Caroline', 'Vaikuntanathan_Vinod', 'Verghese_George', 'Voldman_Joel', 'Ward_Stephen', 'Warde_Cardinal', 'Watts_Michael', 'Weiss_Ron', 'Weiss_Thomas', 'White_Jacob', 'Williams_Virginia', 'Williams_Ryan', 'Willsky_Alan', 'Wilson_Gerald', 'Winston_Patrick', 'Wornell_Gregory', 'Zahn_Markus', 'Zeldovich_Nickolai', 'Zheng_Lizhong', 'Zue_Victor']

faculty_list = ['Bhatia_Sangeeta']
print len(faculty_list)

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
            tokens_stemmed.append(p_stemmer.stem(i).encode('utf8')) #without encode('utf8'), stem appended u' at the start of every word)
        all_abstract.append(tokens_stemmed)
    

print all_abstract
print type(all_abstract)
print len(all_abstract)


import gensim
from gensim import corpora, models
dictionary = corpora.Dictionary(all_abstract)

#print(dictionary.token2id)

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

       



