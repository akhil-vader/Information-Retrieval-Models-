from elasticsearch import Elasticsearch
import elasticsearch.helpers

es = Elasticsearch()

import nltk
import operator
from collections import Counter
from nltk.stem import PorterStemmer
import math

ps = PorterStemmer()

queries={}

f=open('D:\IR-Assignments\Assignments1\AP_DATA\query')
p=open('D:\IR-Assignments\Assignments1\AP_DATA\h')

flist=f.readlines()
f.close()
flist = [line.rstrip('\n') for line in flist]
plist=p.readlines()
p.close()
plist = [line.rstrip('\n') for line in plist]
for i in range(25):
  query_id=flist[i].split()[0]
  query_text=plist[i]
  queries[query_id]=query_text

res_in=es.search(index="apdataset",doc_type="document",body={
  "size": 0,
  "aggs": {
    "vocabSize": {
      "cardinality": {
        "field": "text"
      }
    }
  }
},filter_path=['hits','aggregations'])
D=res_in['hits']['total']
V=res_in['aggregations']['vocabSize']['value']

#######################################-------------MODEL okapi,TF_IDF,okapiBM25---------------------#####################################################################################################

for key,value in queries.items():
  c=Counter()
  tokens = nltk.word_tokenize(value)
  for token in tokens:
    word=ps.stem(token)

    res = elasticsearch.helpers.scan(es, {"_source": True,
                                          "query": {
                                            "match": {
                                              "text": token
                                            }
                                          },
                                          "script_fields": {
                                            "index_df": {
                                              "script": {
                                                "lang": "groovy",
                                                "inline": "_index['text']['" + word + "'].df()"
                                              }
                                            },
                                            "index_tf": {
                                              "script": {
                                                "lang": "groovy",
                                                "inline": "_index['text']['" + word + "'].tf()"
                                              }
                                            }
                                          }
                                          }, index="apdataset", doc_type="document", scroll=u'5m',size=10000,request_timeout=None)
    q=[]
    d={}
    for i in res:
      q.append(i)
    df=q[1]['fields']['index_df'][0]
    for i in range(df):
      tf_w_d = q[i]['fields']['index_tf'][0]
      doc_l = q[i]['_source']['doc_l']
      id = q[i]['_id']
      avg_doc_l = 441
      # okapi_tf = tf_w_d / (tf_w_d + 0.5 + 1.5 * (doc_l / avg_doc_l))
      # tf_idf = (tf_w_d / (tf_w_d + 0.5 + 1.5 * (doc_l / avg_doc_l)))*(math.log(D/df))
      okapi_bm25=math.log(float(D+.5)/float(df+.5))*(float((tf_w_d)+(1.2*tf_w_d))/float(tf_w_d+1.2*((1-.75)+(float(.75*doc_l)/float(avg_doc_l)))))

      # d[id] = okapi_tf
      # d[id] = tf_idf
      d[id] = okapi_bm25
    c=Counter(d)+c
    score=dict(c)
    score_list=[[i,v] for i,v in score.items()]
    score_list.sort(key=operator.itemgetter(1),reverse=True)
    score_list=score_list[:1000]
  for i in range(1000):
    with open('result_okapi_bm25.txt', 'a') as file:
      file.write(str(key) + " " + str("Q0") + " " + str(score_list[i][0]) + " " + str(i+1) + " " + str(score_list[i][1]) + " " + str("Exp")+"\n")




##############################--------MODEL UNIGRAM LM WITH LAPLACE SMOOTHING-------#############################################################



for key,value in queries.items():
  res=es.search(index="apdataset",doc_type="document",body={
    "_source": True,
    "query": {
      "match": {
        "text": value
      }
    },
    "script_fields": {
      "index_df": {
        "script": {
          "lang": "groovy",
          "inline":"_index['text']['"+value+"'].df()"
        }
      },
      "index_tf": {
        "script": {
          "lang": "groovy",
          "inline":"_index['text']['"+value+"'].tf()"
        }
      }
    }
  })
  df=res['hits']['total']
  res2=elasticsearch.helpers.scan(es,{
    "_source": True,
    "query": {
      "match": {
        "text": value
      }
    },
    "script_fields": {
      "index_df": {
        "script": {
          "lang": "groovy",
          "inline":"_index['text']['"+value+"'].df()"
        }
      },
      "index_tf": {
        "script": {
          "lang": "groovy",
          "inline":"_index['text']['"+value+"'].tf()"
        }
      }
    }
  },index="apdataset",doc_type="document",scroll=u'5m')
  q=[]
  d={}
  for i in res2:
    q.append(i)
  for i in range(df):
    doc_l = q[i]['_source']['doc_l']
    id = q[i]['_id']
    p_laplce=math.log(float(1)/float((doc_l+V)))
    d[id]=p_laplce

  c={}
  tokens = nltk.word_tokenize(value)
  for token in tokens:
    word=ps.stem(token)
    res3 = elasticsearch.helpers.scan(es, {"_source": True,
                                           "query": {
                                             "match": {
                                               "text": token
                                             }
                                           },
                                           "script_fields": {
                                             "index_df": {
                                               "script": {
                                                 "lang": "groovy",
                                                 "inline": "_index['text']['" + word + "'].df()"
                                               }
                                             },
                                             "index_tf": {
                                               "script": {
                                                 "lang": "groovy",
                                                 "inline": "_index['text']['" + word + "'].tf()"
                                               }
                                             }
                                           }
                                           }, index="apdataset", doc_type="document", scroll=u'5m')
    q1 = []
    d1 = {}
    for i in res3:
      q1.append(i)
    df1 = q1[1]['fields']['index_df'][0]
    for i in range(df1):
      tf_w_d = q1[i]['fields']['index_tf'][0]
      id = q1[i]['_id']
      doc_l = q[i]['_source']['doc_l']
      p_laplce = math.log(float(tf_w_d + 1) / float((doc_l + V)))
      d1[id] = p_laplce
  #
    g = d.keys() - d1.keys()
    q2 = {}
    for i in g:
      q2[i] = d[i]
    q2.update(d1)
    c={x: q2.get(x, 0) + c.get(x, 0) for x in set(q2).union(c)}
  score_list=[[i,v] for i,v in c.items()]
  score_list.sort(key=operator.itemgetter(1),reverse=True)
  score_list=score_list[:1000]
  for i in range(1000):
    with open('result_lm_lp.txt', 'a') as file:
      file.write(str(key) + " " + str("Q0") + " " + str(score_list[i][0]) + " " + str(i+1) + " " + str(score_list[i][1]) + " " + str("Exp")+"\n")



# # #################################---------MODEL UNIGRAM LM WITH JELINEK_MERCER SMOOTHING----------################################################
#
k=0.7
for key,value in queries.items():
  tokens = nltk.word_tokenize(value)
  c={}
  for token in tokens:
    word=ps.stem(token)
    res=elasticsearch.helpers.scan(es,{
      "_source": True,
      "query": {
        "match": {
          "text": token
        }
      },
      "script_fields": {
        "index_df": {
          "script": {
            "lang": "groovy",
            "inline":"_index['text']['"+word+"'].df()"
          }
        },
        "index_tf": {
          "script": {
            "lang": "groovy",
            "inline":"_index['text']['"+word+"'].tf()"
          }
        }
      }
    },index="apdataset",doc_type="document")
    q=[]
    cf_w=0
    d={}
    for i in res:
      q.append(i)
    df = q[1]['fields']['index_df'][0]
    for i in range(df):
      tf_w_d = q[i]['fields']['index_tf'][0]
      cf_w=tf_w_d+cf_w
    for i in range(df):
      tf_w_d = q[i]['fields']['index_tf'][0]
      id = q[i]['_id']
      doc_l = q[i]['_source']['doc_l']
      lm_jlms=math.log((float(k*tf_w_d)/float(doc_l))+(float((1-k)*cf_w)/float(V)))
      d[id]=lm_jlms
    res2=elasticsearch.helpers.scan(es,{
      "_source": True,
      "query": {
        "match": {
          "text": value
        }
      },
      "script_fields": {
        "index_df": {
          "script": {
            "lang": "groovy",
            "inline":"_index['text']['"+value+"'].df()"
          }
        },
        "index_tf": {
          "script": {
            "lang": "groovy",
            "inline":"_index['text']['"+value+"'].tf()"
          }
        }
      }
    },index="apdataset",doc_type="document",scroll=u'5m')
    q1=[]
    d1={}
    for i in res2:
      q1.append(i)
    for i in range(len(q1)):
      doc_l = q1[i]['_source']['doc_l']
      id = q1[i]['_id']
      lm_jlms = math.log((float((1 - k) * cf_w) /float(V)))
      d1[id]=lm_jlms
    g = d1.keys() - d.keys()
    q2 = {}
    for i in g:
      q2[i] = d1[i]
    q2.update(d)
    c={x: q2.get(x, 0) + c.get(x, 0) for x in set(q2).union(c)}
  score_list=[[i,v] for i,v in c.items()]
  score_list.sort(key=operator.itemgetter(1),reverse=True)
  score_list=score_list[:1000]
  for i in range(1000):
    with open('result_lm_jlm.txt', 'a') as file:
      file.write(str(key) + " " + str("Q0") + " " + str(score_list[i][0]) + " " + str(i + 1) + " " + str(score_list[i][1]) + " " + str("Exp") + "\n")









