import json
import os
from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch
es = Elasticsearch()

dir="D:\IR-Assignments\Assignments1\AP_DATA\\ap89_collection"
list_of_files =os.listdir(dir)

for filename in list_of_files:
    filetext=open(dir+"\\"+filename)

    filesoup = BeautifulSoup(filetext.read(), "lxml")
    doc_elements = filesoup.findAll("doc")
    for doc_element in doc_elements:
        docno = doc_element.find("docno").text.strip()
        text_elements = doc_element.find_all("text")
        text = ""
        for text_element in text_elements:
            text += text_element.text.strip()+ " "
            doc_l= len(text.split())
        data = {
            'docno': docno,
            'text': text,
            'doc_l': doc_l}
        json_docs=json.dumps(data)
        res=es.index(index="apdataset",doc_type="document",id=docno,body=json_docs,ignore=[400,404])
        print(res['created'])