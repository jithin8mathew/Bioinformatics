# PYTHON CODE TO PARSE THE JASPAR DATABASE OF REGULATORY MOTIFS FROM A LIST OF SEQUENCE ID'S AND OBTAIN INFORMATION USING JASPAR API

import json
import requests, sys

w=open("Path_to_the_accession_ID_list\\large_motif_list.txt","r+").readlines()
ID=[]
clss=[]

def parse_jaspar(seq_id):
	query_url="http://jaspar.genereg.net/api/v1/matrix/"+seq_id+"/"
	 
	result = requests.get(query_url)
	 
	if not result.ok:
		r.raise_for_status()
		sys.exit()
	 
	decoded = result.json()
	#print(repr(decoded))

	for x , y in decoded.items():
		print(x,"\t", y)
		if x == "matrix_id":
			[ID.append(y)]
		if x == "class":
			[clss.append(ite) for ite in y]
		
for s in w:
	s=str(s.rstrip("\n"))
	parse_jaspar(s)

dct=dict(zip(ID,clss))
print(dct)

