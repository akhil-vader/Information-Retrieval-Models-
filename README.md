# Information-Retrieval-Models-

Objective:
Implement and compare various retrieval systems using vector space models and language models.

This involves writing two programs:
A program to parse the corpus and index it with elasticsearch
A query processor, which runs queries from an input file using a selected retrieval model

First Steps
Download and install elasticsearch, and the kibana plugin
Document Indexing
Create an index of the downloaded corpus. You will need to write a program to parse the documents and send them to your elasticsearch.

The corpus files are in a standard format used by TREC. Each file contains multiple documents. The format is similar to XML, but standard XML and HTML parsers will not work correctly. Instead, read the file one line at a time with the following rules:

Each document begins with a line containing <DOC> and ends with a line containing </DOC>.
The first several lines of a document’s record contain various metadata. Read the <DOCNO> field and used it as the ID of the document.
The document contents are between lines containing <TEXT> and </TEXT>.
All other file contents were ignored.
Make sure to index term positions: we will need them later. You are also free to add any other fields to your index for later use. This might be the easiest way to get particular values used in the scoring functions. However, if a value is provided by elasticsearch then its easy to retrieve it using the elasticsearch API rather than calculating and storing it yourself.

Query execution
Write program to run the queries in the file query_desc.51-100.short.txt, included in the data .zip file. It will run all queries (omitting the leading number) using each of the retrieval models listed below, and output the top 100 results for each query to an output file. If a particular query has fewer than 100 documents with a nonzero matching score, then whichever documents having nonzero scores were listed.

Precisely one output file per retrieval model. Each line of an output file specifies one retrieved document, in the following format:

<query-number> Q0 <docno> <rank> <score> Exp
Where:

<query-number> is the number preceding the query in the query list
<docno> is the document number, from the <DOCNO> field (which we asked you to index)
<rank> is the document rank: an integer from 1-1000
<score >is the retrieval model’s matching score for the document
Q0 and Exp are entered literally
  
The program will run queries against elasticsearch. Instead of using their built in query engine, we will be retrieving information such as TF and DF scores from elasticsearch and implementing our own document ranking. It will be helpful if you write a method which takes a term as a parameter and retrieves the postings for that term from elasticsearch. You can then easily reuse this method to implement the retrieval models.

Implemented the following retrieval models, using TF and DF scores from elasticsearch index, as needed.

ES built-in
Use ES query with the API "match"{"body_text":"query keywords"}. This should be somewhat similar to BM25 scoring

Okapi TF
This is a vector space model using a slightly modified version of TF to score documents. The Okapi TF score for term ww in document dd is as follows.

okapi_tf(w,d)=tfw,dtfw,d+0.5+1.5⋅(len(d)/avg(len(d)))
okapi_tf(w,d)=tfw,dtfw,d+0.5+1.5⋅(len(d)/avg(len(d)))

Where:
tfw is the term frequency of term ww in document dd
len(d) is the length of document dd
avg(len(d)) is the average document length for the entire corpus
The matching score for document dd and query qq is as follows.

tf(d,q)=∑w∈qokapi_tf(w,d)
tf(d,q)=∑w∈qokapi_tf(w,d)

TF-IDF
This is the second vector space model. The scoring function is as follows.

tfidf(d,q)=∑w∈qokapi_tf(w,d)⋅logDdfw
tfidf(d,q)=∑w∈qokapi_tf(w,d)⋅log⁡Ddfw

Where:
D is the total number of documents in the corpus
dfw is the number of documents which contain term ww

Okapi BM25
BM25 is a language model based on a binary independence model. Its matching score is as follows.

bm25(d,q)=∑w∈q[log(D+0.5dfw+0.5)⋅tfw,d+k1⋅tfw,dtfw,d+k1((1−b)+b⋅len(d)avg(len(d)))⋅tfw,q+k2⋅tfw,qtfw,q+k2]
bm25(d,q)=∑w∈q[log⁡(D+0.5dfw+0.5)⋅tfw,d+k1⋅tfw,dtfw,d+k1((1−b)+b⋅len(d)avg(len(d)))⋅tfw,q+k2⋅tfw,qtfw,q+k2]

Where:
tfw,qtfw,q is the term frequency of term ww in query qq
k1,k2, and b are constants. You try your own values.

Unigram LM with Laplace smoothing
This is a language model with Laplace (“add-one”) smoothing. We will use maximum likelihood estimates of the query based on a multinomial model “trained” on the document. The matching score is as follows.

lm_laplace(d,q)=∑w∈qlogp_laplace(w|d)p_laplace(w|d)=tfw,d+1len(d)+V
lm_laplace(d,q)=∑w∈qlog⁡p_laplace(w|d)p_laplace(w|d)=tfw,d+1len(d)+V

Where:
V is the vocabulary size – the total number of unique terms in the collection.

Unigram LM with Jelinek-Mercer smoothing
This is a similar language model, except that here we smooth a foreground document language model with a background model from the entire corpus.

lm_jm(d,q)=∑w∈qlogp_jm(w|d)p_jm(w|d)=λtfw,dlen(d)+(1−λ)∑d′tfw,d′∑d′len(d′)
lm_jm(d,q)=∑w∈qlog⁡p_jm(w|d)p_jm(w|d)=λtfw,dlen(d)+(1−λ)∑d′tfw,d′∑d′len(d′)

Where:
λ∈(0,1) is a smoothing parameter which specifies the mixture of the foreground and background distributions.

Evaluation
A) Compare manually the top 10 docs returned by ESBuilt-In, TFIDF, BM25, LMJelinek, for 5 queries specified by TAs.

B) Download trec_eval and use it to evalute your results for each retrieval model.

To perform an evaluation, run:

trec_eval [-q] qrel_file results_file
The -q option shows a summary average evaluation across all queries, followed by individual evaluation results for each query; without the -q option, you will see only the summary average. The trec_eval program provides a wealth of statistics about how well the uploaded file did for those queries, including average precision, precision at various recall cut-offs, and so on.

Created a document showing the following metrics:

The uninterpolated mean average precision numbers for all six retrieval models.
Precision at 10 and precision at 30 documents for all six retrieval models.
Be prepared to answer questions during your demo as to how model performance compares, why the best model outperformed the others, and so on.
