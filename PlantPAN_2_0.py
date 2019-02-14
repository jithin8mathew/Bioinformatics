#Script for processing PlantPAN 2.0 promoter site analysis or transcription factor binding output
data=open('Path to CSV file','r+').readlines()
sections=[]
TF=[]
for x in data:
	x=x.split(',')
	sections.append(x[0])
	TF.append(x[6])
sections=set(sections)

for items in sections:
	dct={}
	lst=[]
	for elem in data:
		elem=elem.split(',')
		if elem[0]==items:
			lst.append(elem[6])
	for x in lst:
		if x in dct:
			dct[x]+=1
		else:
			dct[x]=1
	print('*'*50)
	print(items)
	print(dct)
	print('*'*50)
