from scholarly import scholarly
import yaml

# something like ssh -D 9050 -q -C -N dlange@lxplus747.cern.ch
from scholarly import scholarly, ProxyGenerator
# default values are shown below
proxies = {'http' : 'socks5://127.0.0.1:9050', 'https': 'socks5://127.0.0.1:9050'}
pg = ProxyGenerator()
pg.SingleProxy(**proxies)

scholarly.use_proxy(pg)

# Retrieve the author's data, fill-in, and print
#author=scholarly.search_author_id('4poYWhEAAAAJ')
search_query = scholarly.search_author('Vassil Vassilev')

while True:
    print("Iter")
    try:
        author = next(search_query).fill()
        if 'cern' in author.email: break
        #print(author)
    except StopIteration:
        break
#sys.exit(1)
print(author)

print("Titles")
# Print the titles of the author's publications
stuff=[]
for pub in author.publications:
    pub.fill()
    this_stuff={}
    for key in pub.bib.keys():
        this_stuff[key]=pub.bib[key]
    stuff.append(this_stuff)

ff=open('vassil.yaml','w')
yaml.dump(stuff,ff,allow_unicode=True)
ff.close()
